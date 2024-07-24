import torchvision
import torch.nn as nn
import torch
from torch import optim
from helpers import NetworkMix , show_time, general_num_params, Clock
import logging, sys, os
from sklearn.metrics import accuracy_score
import numpy as np
import time

total_runtime_hours = 24
total_runtime_seconds = total_runtime_hours * 60 * 60
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join( 'Searchlog.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
class NAS:
    """
    ====================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The NAS class will receive the following inputs
        * train_loader: The train loader created by your DataProcessor
        * valid_loader: The valid loader created by your DataProcessor
        * metadata: A dictionary with information about this dataset, with the following keys:
            'num_classes' : The number of output classes in the classification problem
            'codename' : A unique string that represents this dataset
            'input_shape': A tuple describing [n_total_datapoints, channel, height, width] of the input data
            'time_remaining': The amount of compute time left for your submission
            plus anything else you added in the DataProcessor

        You can modify or add anything into the metadata that you wish,
        if you want to pass messages between your classes,
    """
    def __init__(self, train_loader, valid_loader,  metadata):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        

    def train(self,epochs,model):
        self.model = model
        self.epochs = epochs
        self.optimizer = optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=3e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        if torch.cuda.is_available():
            self.model.cuda()
        t_start = time.time()
        for epoch in range(epochs):
            self.model.train()
            labels, predictions = [], []
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model.forward(data)

                # store labels and predictions to compute accuracy
                labels += target.cpu().tolist()
                predictions += torch.argmax(output, 1).detach().cpu().tolist()

                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

            train_acc = accuracy_score(labels, predictions)
            valid_acc = self.evaluate()
            logging.info("\tEpoch {:>3}/{:<3} | Train Acc: {:>6.2f}% | Valid Acc: {:>6.2f}% | T/Epoch: {:<7} |".format(
                epoch + 1, self.epochs,
                train_acc * 100, valid_acc * 100,
                show_time((time.time() - t_start) / (epoch + 1))
            ))
        logging.info("  Total runtime: {}".format(show_time(time.time() - t_start)))
        return train_acc*100, valid_acc*100


    def evaluate(self):
        self.model.eval()
        labels, predictions = [], []
        for data, target in self.valid_loader:
            data = data.to(self.device)
            output = self.model.forward(data)
            labels += target.cpu().tolist()
            predictions += torch.argmax(output, 1).detach().cpu().tolist()
        return accuracy_score(labels, predictions)


    def search_depth_and_width(self):

        

        logging.info('#############################################################################')
        logging.info('INITIALIZING DEPTH AND WIDTH SEARCH...')

        runclock = Clock(total_runtime_seconds)

        target_acc= 100
        min_width= 16
        max_width= 128
        width_resolution = 16
        min_depth= 5
        max_depth= 100
        ch_drop_tolerance = 0.05
        target_acc_tolerance = 0.10
        channels = 16 
        layers = min_depth
        f_epochs = 0
        ch_break_tolerance = 3
        dp_break_tolerance = 1
        dp_add_tolerance = 0.10
        add_epochs_w = 0
        add_epochs = 0
        epochs = 5

        # Initialize
        curr_arch_ops = next_arch_ops = np.zeros((layers,), dtype=int)
        curr_arch_kernel = next_arch_kernel = 3*np.ones((layers,), dtype=int)

        curr_arch_train_acc = next_arch_train_acc = 0.0
        curr_arch_test_acc = next_arch_test_acc = 0.0
        logging.info('RUNNING MACRO SEARCH FIRST on the dataset %s', self.metadata['codename'])

        model = NetworkMix(channels,self.metadata, layers, curr_arch_ops,
                curr_arch_kernel)  
        
        logging.info('MODEL DETAILS')
        logging.info("Model Depth %s Model Width %s", layers, channels)
        logging.info("Model Layers %s Model Kernels %s", curr_arch_ops, curr_arch_kernel)
        logging.info('Training epochs %s', epochs)
        logging.info("Model Parameters = %f", general_num_params(model))
        logging.info('Training Model...')
        # train model using your Trainer
        print("\n=== Training ===")
        print("  Allotted compute time remaining: ~{}".format(show_time(runclock.check())))
        curr_arch_train_acc, curr_arch_test_acc  = self.train(epochs, model)

        logging.info("Baseline Train Acc %f Baseline Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)

        # Search depth
        depth_fail_count = 0
        channels_up = False
        while ((curr_arch_test_acc < (target_acc - target_acc_tolerance)) and (layers != max_depth)):
            
            # The possibility exists if trained for too long.
            if (curr_arch_train_acc == 99.5):
                break;  
            
            else:
            # prepare next candidate architecture.  
                layers += 1
            next_arch_ops = np.zeros((layers,), dtype=int)
            next_arch_kernel = 3*np.ones((layers,), dtype=int)
            model = NetworkMix(channels,self.metadata, layers, next_arch_ops,
                next_arch_kernel)  
                        
            logging.info('#############################################################################')
            logging.info('Moving to Next Candidate Architecture...')
            logging.info('MODEL DETAILS')
            logging.info("Model Depth %s Model Width %s", layers, channels)
            logging.info("Model Layers %s Model Kernels %s", next_arch_ops, next_arch_kernel)
            logging.info('Total number of epochs %s', epochs)
            logging.info("Model Parameters = %f", general_num_params(model))
            logging.info("Depth Fail Count %s", depth_fail_count)
            logging.info('Training Model...')
            # train model using your Trainer
            print("\n=== Training ===")
            print("  Allotted compute time remaining: ~{}".format(show_time(runclock.check())))
            next_arch_train_acc, next_arch_test_acc  = self.train(epochs,model)

            logging.info("Candidate Train Acc %f Candidate Val Acc %f", next_arch_train_acc, next_arch_test_acc)
            
            # As long as we get significant improvement by increasing depth.
            
            if (next_arch_test_acc >= curr_arch_test_acc + dp_add_tolerance):
                # update current architecture.
                depth_fail_count = 0
                curr_arch_ops = next_arch_ops
                curr_arch_kernel = next_arch_kernel
                logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
                curr_arch_train_acc = next_arch_train_acc
                curr_arch_test_acc = next_arch_test_acc
                f_channels = channels
                f_epochs = epochs
                logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)
                                
            elif((next_arch_test_acc < curr_arch_test_acc + dp_add_tolerance) and ((depth_fail_count != dp_break_tolerance))):
                depth_fail_count += 1
                # layers -= 1
                # epochs = epochs + add_epochs
                logging.info('Increasing Epoch in DEPTH block...')
                logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)
                logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
                continue
                
            elif(channels != max_width):
                if not channels_up:
                    layers -= 1
                    channels += int(width_resolution/2)
                    channels_up = True
                    logging.info('Increasing CHANNELS in WIDTH block...')
                else: 
                    logging.info('Increasing Epoch in WIDTH block...')
                    epochs = epochs + add_epochs
                    channels_up = False
                    # layers -= 1
                        
                logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)
                logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
            else:
                logging.info('INCREASING CHANNELS REPEAT...')
                logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)
                logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
                break
        # Search width
        # During width search lenght of curr_arch_ops and curr_arch_kernel shall not change but only channels.

        f_layers = len(curr_arch_ops) # discovered final number of layers
        logging.info('Discovered Final Depth %s', f_layers)
        logging.info('Epochs so far %s', f_epochs)
        logging.info('END OF DEPTH SEARCH...')
        best_arch_test_acc = curr_arch_test_acc
        best_arch_train_acc = curr_arch_train_acc
        logging.info('#############################################################################')
        logging.info('#############################################################################')
        logging.info('')
        logging.info('RUNNING WIDTH SEARCH NOW...') 

        channels = f_channels
        width_fail_count = 0
        while (channels > min_width):
            # prepare next candidate architecture.
            channels = channels - int(width_resolution/4)
            # Although these do not change.
            model = NetworkMix(channels,self.metadata, f_layers, curr_arch_ops,
                curr_arch_kernel)      
            epochs = epochs + add_epochs_w

            logging.info('Moving to Next Candidate Architecture...')
            logging.info('MODEL DETAILS')
            logging.info("Model Depth %s Model Width %s", f_layers, channels)
            logging.info("Model Layers %s Model Kernels %s", curr_arch_ops, curr_arch_kernel)
            logging.info('Total number of epochs %f', epochs)
            logging.info("Model Parameters = %f", general_num_params(model))
            logging.info('Training Model...')
            logging.info("Width Fail Count %s", width_fail_count)
            # train and test candidate architecture.
            # train model using your Trainer
            print("\n=== Training ===")
            print("  Allotted compute time remaining: ~{}".format(show_time(runclock.check())))
            next_arch_train_acc, next_arch_test_acc  = self.train(epochs, model)

            logging.info("Candidate Train Acc %f Candidate Val Acc %f", next_arch_train_acc, next_arch_test_acc)

            if (next_arch_test_acc >= (curr_arch_test_acc - 0.0)):

                logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
                curr_arch_train_acc = next_arch_train_acc
                curr_arch_test_acc = next_arch_test_acc
             
                f_channels = channels 
                f_epochs = epochs
                logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)
                width_fail_count = 0
            elif (width_fail_count != ch_break_tolerance):
                width_fail_count += 1
                logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
                logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)

                continue
            else:
                logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
                logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)

                break; 
                
        logging.info('Discovered Final Depth %s', f_layers)
        logging.info('Discovered Final Width %s', f_channels)
        logging.info('Discovered Final Epochs %s', f_epochs)
        logging.info('END OF WIDTH SEARCH...')  
        logging.info('#############################################################################')
        logging.info('#############################################################################')
        logging.info('')  

        return curr_arch_ops, curr_arch_kernel, f_channels, f_layers

    """
    ====================================================================================================================
    SEARCH =============================================================================================================
    ====================================================================================================================
    The search function is called with no arguments, and expects a PyTorch model as output. Use this to
    perform your architecture search. 
    """
    def search(self):
       
        curr_arch_ops, curr_arch_kernel, f_channels, f_layers = self.search_depth_and_width()
        # curr_arch_ops = [0, 0, 0 ,0, 0 ,0, 0]
        # curr_arch_kernel = [3, 3, 3, 3, 3, 3, 3]
        # f_channels = 16
        # f_layers = 7
        model = NetworkMix(f_channels, self.metadata, f_layers, curr_arch_ops,
         curr_arch_kernel)
        return model