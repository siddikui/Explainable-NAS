import torchvision
import torch.nn as nn
import torch
from torch import optim
from helpers import NetworkMix , show_time, general_num_params, Clock, set_seed
import logging, sys, os
from sklearn.metrics import accuracy_score
import numpy as np
import time
import copy

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
    def __init__(self, train_loader, valid_loader, metadata):

        total_train_size = len(train_loader.dataset)
        print(f"Total train size: {total_train_size}")

        # Split the training dataset into two equal subsets
        train_subset_size = total_train_size // 16
        train_subset, _ = torch.utils.data.random_split(
            train_loader.dataset, [train_subset_size, total_train_size - train_subset_size]
        )

        # DataLoader for subset of the training dataset
        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=train_loader.batch_size,
            drop_last=True,
            shuffle=True
        )

        total_valid_size = len(valid_loader.dataset)
        print(f"Total valid size: {total_valid_size}")

        # Split the validation dataset into two equal subsets
        valid_subset_size = total_valid_size // 16
        valid_subset, _ = torch.utils.data.random_split(
            valid_loader.dataset, [valid_subset_size, total_valid_size - valid_subset_size]
        )

        # DataLoader for subset of the validation dataset
        valid_loader = torch.utils.data.DataLoader(
            valid_subset,
            batch_size=valid_loader.batch_size,
            drop_last=True,
            shuffle=True
        )

        print(f"Train loader size after split: {len(train_loader.dataset)}")
        print(f"Valid loader size after split: {len(valid_loader.dataset)}") 

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        set_seed(1)


    
    def train(self, epochs, model):
        self.model = model
        self.epochs = epochs
        self.optimizer = optim.SGD(model.parameters(), lr=.025, momentum=.9, weight_decay=3e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

        if torch.cuda.is_available():
            self.model.cuda()

        t_start = time.time()
        best_valid_acc = 0.0
        best_train_acc = 0.0
        best_epoch = 0
        best_model = None

        for epoch in range(epochs):
            self.model.train()
            labels, predictions = [], []
            
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model.forward(data)

                # Store labels and predictions to compute accuracy
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

            # Check if this is the best model based on validation accuracy
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_train_acc = train_acc
                self.best_epoch = epoch + 1
                best_model = copy.deepcopy(self.model)  # Keep a copy of the best model

        self.model = best_model
        logging.info("Candidate Evaluation Time: {}".format(show_time(time.time() - t_start)))
        
        # Return best train acc, best valid acc, and the best epoch number
        return best_train_acc * 100, best_valid_acc * 100


    def evaluate(self):
        self.model.eval()
        labels, predictions = [], []
        for data, target in self.valid_loader:
            data = data.to(self.device)
            output = self.model.forward(data)
            labels += target.cpu().tolist()
            predictions += torch.argmax(output, 1).detach().cpu().tolist()
        return accuracy_score(labels, predictions)

    def save_checkpoint(self, model, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.metadata["codename"]+".pth")
        print(f"Checkpoint saved to {self.metadata['codename']}.pth")

    def search_depth_and_width(self):
        

        runclock = Clock(total_runtime_seconds)

        target_acc= 100
        min_width= 16
        max_width= 160
        width_resolution = 8
        min_depth= 5
        max_depth= 40
        max_epochs = 50

        channels = f_channels = 16 
        layers = min_depth
        
        add_epochs = 1
        epochs = 1
        f_epochs = 0

        macro_count = 0

        # Initialize
        curr_arch_ops = next_arch_ops = np.zeros((layers,), dtype=int)
        curr_arch_kernel = next_arch_kernel = 3*np.ones((layers,), dtype=int)
        curr_arch_train_acc = next_arch_train_acc = 0.0
        curr_arch_test_acc = next_arch_test_acc = 0.0

        logging.info('RUNNING SEARCH on %s', self.metadata['codename'])
        model = NetworkMix(channels,self.metadata, layers, curr_arch_ops, curr_arch_kernel)          
        logging.info("Model Depth %s Model Width %s Train Epochs %s", layers, channels, epochs)

        logging.info("Model Parameters = %f", general_num_params(model))
        logging.info('Evaluating Candidate Model...')

        curr_arch_train_acc, curr_arch_test_acc  = self.train(epochs, model)

        logging.info("Baseline Train Acc %f Baseline Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)

        # SEARCH MODEL
        
        layers_up = channels_up = epochs_up = True


        while (curr_arch_test_acc < target_acc):
            torch.cuda.empty_cache()

            
            # The possibility exists if trained for too long.
            if (curr_arch_train_acc >= 99):
                break;  
            
            # prepare next candidate architecture.  

            if ((layers >= max_depth and channels >=max_width) or epochs >= max_epochs):
                break;

            if layers_up and layers < max_depth: 
                
                layers += 1
                layers_up = False
                channels_up = True
                epochs_up = True
                macro_count+=1
            
            elif channels_up and channels < max_width:
                
                channels += int(width_resolution/2)
                channels_up = False
                epochs_up = True 
                macro_count+=1

            elif epochs_up and epochs < max_epochs:
                                
                epochs = epochs + add_epochs
                epochs_up = False
                layers_up = True
                channels_up = True
                #if macro_count%4==0:
                #    epochs = epochs - add_epochs
                #    layers += 1
                #    channels += int(width_resolution/2)


            next_arch_ops = np.zeros((layers,), dtype=int)
            next_arch_kernel = 3*np.ones((layers,), dtype=int)
            model = NetworkMix(channels,self.metadata, layers, next_arch_ops, next_arch_kernel)  
                        
            logging.info('#############################################################################')
            logging.info('Moving to Next Candidate Architecture...')
            logging.info("Model Depth %s Model Width %s Train Epochs %s", layers, channels, epochs)
            logging.info("Model Parameters = %f", general_num_params(model))

            if general_num_params(model) > 5_000_000:
                logging.info("Model Parameters Exceed Upper Bound")
                break;

            nas_time_secs = 120
            if runclock.check() < (24*60*60-nas_time_secs):
                print("Allotted compute time remaining: ~{}".format(show_time(runclock.check())))
                logging.info("Time Limit Exceeds Given Deadline")
                break;

            #logging.info("Allotted compute time remaining: ~{}".format(show_time(runclock.check())))
            logging.info("Allotted compute time remaining: ~{}".format(runclock.check()))

            logging.info('Evaluating Candidate Model...')
            
            next_arch_train_acc, next_arch_test_acc  = self.train(epochs,model)

            logging.info("Candidate Train Acc %f Candidate Val Acc %f", next_arch_train_acc, next_arch_test_acc)
            
            # As long as we get significant improvement by increasing depth.
            
            if (next_arch_test_acc > curr_arch_test_acc + 0.5):

                if layers_up is False:
                    if channels_up is False:
                        channels_up = True
                    else:
                        layers_up =True    

                # update current architecture.
                curr_arch_ops = next_arch_ops
                curr_arch_kernel = next_arch_kernel
                logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
                curr_arch_train_acc = next_arch_train_acc
                curr_arch_test_acc = next_arch_test_acc
                f_channels = channels
                f_epochs = epochs
                s_epoch = self.best_epoch
                self.save_checkpoint(model, self.best_epoch)

                logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)                        
                logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
            else:
                logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)                        
                logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
                continue
        # Search width
        # During width search lenght of curr_arch_ops and curr_arch_kernel shall not change but only channels.

        f_layers = len(curr_arch_ops) # discovered final number of layers


                
        logging.info('Discovered Final Depth %s', f_layers)
        logging.info('Discovered Final Width %s', f_channels)
        logging.info('Discovered Final Epochs %s best saved epoch %s', f_epochs, s_epoch)
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