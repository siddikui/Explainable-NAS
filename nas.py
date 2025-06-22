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
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from collections import Counter


total_runtime_hours = 24
total_runtime_seconds = total_runtime_hours * 60 * 60


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('Challenge257.txt'))
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
    
       
        search_size = 0.25
        
        #print(type(train_loader.dataset))  # Check the dataset type
        #print(dir(train_loader.dataset))   # List all attributes and methods

        data = train_loader.dataset.x  # The data samples
        labels = train_loader.dataset.y
        
        train_indices, val_indices = train_test_split(range(len(labels)),
                                                      test_size=1-search_size,  # 25% of the data
                                                      stratify=labels#,  # Preserve class distribution
                                                      #random_state=2  # For reproducibility
                                                      )
                                                      

                                                      
        train_subset = Subset(train_loader.dataset, train_indices)
        val_subset = Subset(train_loader.dataset, val_indices) 
        
        train_loader = torch.utils.data.DataLoader(train_subset, 
                                                          batch_size=64, 
                                                          drop_last=True,
                                                          shuffle=True)
                                                          
        val_loader_subset = torch.utils.data.DataLoader(val_subset, batch_size=64, shuffle=False)       
        


        
        ########################################################################################################
        '''
        #Old train split for search
        total_train_size = len(train_loader.dataset)
        print(f"Total train size: {total_train_size}")

        # Split the training dataset into two equal subsets
        train_subset_size = total_train_size // 4

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
        '''
        total_valid_size = len(valid_loader.dataset)
        print(f"Total valid size: {total_valid_size}")

        # Split the validation dataset into two equal subsets
        valid_subset_size = total_valid_size // 1

        valid_subset, _ = torch.utils.data.random_split(
            valid_loader.dataset, [valid_subset_size, total_valid_size - valid_subset_size]
        )

        # DataLoader for subset of the validation dataset
        valid_loader = torch.utils.data.DataLoader(
            valid_subset,
            batch_size=64,
            drop_last=True,
            shuffle=False
        )

        print(f"Train loader size after split: {len(train_loader.dataset)}")
        print(f"Valid loader size after split: {len(valid_loader.dataset)}") 
        
        ########################################################################################################
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')


    

    
    def train(self, epochs, model):
        self.model = model
        self.epochs = epochs
        self.optimizer = optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=3e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

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
                #data, target = data.to(self.device), target.to(self.device)
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                #print(data.dtype, target.dtype)
                target=target.to(torch.int64)
                self.optimizer.zero_grad()
                output = self.model.forward(data)

                # Store labels and predictions to compute accuracy
                labels += target.cpu().tolist()
                predictions += torch.argmax(output, 1).detach().cpu().tolist()

                loss = self.criterion(output, target)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()

            

            train_acc = accuracy_score(labels, predictions)
            valid_acc = self.evaluate()
            
            logging.info("\tEpoch {:>3}/{:<3} | Train Acc: {:>6.2f}% | Valid Acc: {:>6.2f}% | T/Epoch: {:<7} | LR: {:>2.6f} |".format(
                epoch + 1, self.epochs,
                train_acc * 100, valid_acc * 100,
                show_time((time.time() - t_start) / (epoch + 1)),
                self.scheduler.get_last_lr()[0]
            ))
            #self.scheduler.step()
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


    def search_depth_and_width(self):

        logging.info('RUNNING SEARCH on %s', self.metadata['codename'])

        runclock = Clock(total_runtime_seconds)

        nas_time_secs = 3600*3
        max_params = 3_000_000

        target_acc= 100
        min_width=  16
        max_width= 128
        width_resolution = 32
        depth_resolution = 4
        min_depth= 8
        max_depth= 100
        max_epochs = 10

        max_models = 10
        Rand_train = 10
        candidate_count = 0

        channels = f_channels = min_width#16 
        layers = min_depth
        
        add_epochs = 1# 1
        s_epoch = epochs = 1#1
        f_epochs = 0

        macro_count = 0

        bst_dep = []
        bst_wdt = []
        bst_epc = []
        bst_tac = []
        bst_vac = []
        bst_prm = []




        # Baseline Model 
        curr_arch_ops = next_arch_ops = np.zeros((layers,), dtype=int)
        curr_arch_kernel = next_arch_kernel = 3*np.ones((layers,), dtype=int)
        curr_arch_train_acc = next_arch_train_acc = 0.0
        curr_arch_test_acc = next_arch_test_acc = 0.0

        model = NetworkMix(channels,self.metadata, layers, curr_arch_ops, curr_arch_kernel)          
        logging.info("Model Depth %s Model Width %s Train Epochs %s", layers, channels, epochs)
        logging.info("Model Parameters = %f", general_num_params(model))
        #logging.info("Model = %f", model)

        logging.info('Evaluating Baseline Model...')

        curr_arch_train_acc, curr_arch_test_acc  = self.train(epochs, model)
        
        self.save_checkpoint(model, epochs) # Baseline model

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
            #if epochs >= max_epochs:    
                break;

            if layers_up and layers < max_depth: 
                
                layers += depth_resolution
                channels += int(width_resolution/2)
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



            logging.info('#############################################################################')
            logging.info('Moving to Next Candidate Architecture...')
            logging.info("Model Depth %s Model Width %s Train Epochs %s", layers, channels, epochs)

            archbestt, archbestv = 0.0, 0.0
            next_arch_ops = np.zeros((layers,), dtype=int)
            next_arch_kernel = 3*np.ones((layers,), dtype=int)

            model = NetworkMix(channels,self.metadata, layers, next_arch_ops, next_arch_kernel)                  
            logging.info("Model Parameters = %f", general_num_params(model))
            #logging.info("Model = %f", model)
            if general_num_params(model) > max_params:
                logging.info("Model Parameters Exceed Upper Bound")
                break;
            candidate_count += 1
            logging.info('Candidate count: %f',candidate_count)  
            
            if candidate_count > max_models:
                logging.info("Maximum Models Evaluated")
                break;              
            # Training same candidate with multiple initializations.    
            for i in range(Rand_train):
                set_seed(i)
                torch.cuda.empty_cache()

                logging.info("INITIALIZING RUNNUNG RUN %f", i) 
                model = NetworkMix(channels,self.metadata, layers, next_arch_ops, next_arch_kernel)             

                next_arch_train_acc, next_arch_test_acc  = self.train(epochs,model)

                if next_arch_test_acc > archbestv:
                    archbestt = next_arch_train_acc
                    archbestv = next_arch_test_acc
                    model = self.model

                logging.info("Candidate Train Acc %f Candidate Val Acc %f", next_arch_train_acc, next_arch_test_acc)
            
            # As long as we get significant improvement by increasing depth.
            
            next_arch_train_acc = archbestt
            next_arch_test_acc = archbestv
            logging.info("Candidate Best Train %f Candidate Best Val %f", next_arch_train_acc, next_arch_test_acc)

            if (next_arch_test_acc > curr_arch_test_acc + 0.20):

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
                #self.save_checkpoint(model, epochs)

                logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)                        
                logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
                
                bst_dep.append(layers)
                bst_wdt.append(channels)
                bst_epc.append(epochs)
                bst_tac.append(round(next_arch_train_acc,2))
                bst_vac.append(round(next_arch_test_acc,2))
                bst_prm.append(general_num_params(model))

                logging.info('Best Arch Layers: %s', bst_dep)
                logging.info('Best Arch Channels: %s', bst_wdt)
                logging.info('Best Arch Epochs: %s', bst_epc)
                logging.info('Best Arch Train Acc: %s', bst_tac)
                logging.info('Best Arch Val Acc: %s', bst_vac)
                logging.info('Best Arch Params: %s', bst_prm)



            else:
                logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)                        
                logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
                continue
        # Search width
        # During width search lenght of curr_arch_ops and curr_arch_kernel shall not change but only channels.

        f_layers = len(curr_arch_ops) # discovered final number of layers


        logging.info('Discovered Depth %s', f_layers)
        logging.info('Discovered Width %s', f_channels)
        logging.info('Discovered Epochs %s best saved epoch %s', f_epochs, s_epoch)


        logging.info('#############################################################################')
        logging.info('#############################################################################')
        logging.info('Best Arch Layers: %s', bst_dep)
        logging.info('Best Arch Channels: %s', bst_wdt)
        logging.info('Best Arch Epochs: %s', bst_epc)
        logging.info('Best Arch Train Acc: %s', bst_tac)
        logging.info('Best Arch Val Acc: %s', bst_vac)
        logging.info('Best Arch Params: %s', bst_prm)
        logging.info('#############################################################################')
        logging.info('#############################################################################')
        logging.info('')  

        ###################################################################################
        ###################################################################################
        ########### HYPER PARAMETER SEARCH BEGINS ############################


        #max_models = len(bst_dep) * 2
        candidate_count = 0
        while len(bst_dep) > 1:


            bst_dep_2 = []
            bst_wdt_2 = []
            bst_epc_2 = []
            bst_tac_2 = []
            bst_vac_2 = []
            bst_prm_2 = []
            
            if epochs == max_epochs+1:
                break

            if candidate_count > max_models:
                break
            for i in range(len(bst_dep)):
                layers = bst_dep[len(bst_dep)-2-i]
                channels = bst_wdt[len(bst_wdt)-2-i]

                if layers <= 8:
                    pass

                if i == 0:
                    epochs = max(f_epochs,bst_epc[len(bst_epc)-1]) + 1

                else:
                    #epochs = bst_epc[len(bst_epc)-1] + i + 1
                    epochs = epochs + 1
                
                archbestt, archbestv = 0.0, 0.0                

                next_arch_ops = np.zeros((layers,), dtype=int)
                next_arch_kernel = 3*np.ones((layers,), dtype=int)
                model = NetworkMix(channels,self.metadata, layers, next_arch_ops, next_arch_kernel)                             
                
                logging.info('Moving to Next Candidate Architecture...')
                logging.info("Model Depth %s Model Width %s Train Epochs %s", layers, channels, epochs)
                logging.info("Model Parameters = %f", general_num_params(model))
                #logging.info("Model = %f", model)


                for i in range(Rand_train):
                    set_seed(i+10)
                    torch.cuda.empty_cache()

                    logging.info("INITIALIZING RUNNUNG RUN %f", i) 
                    model = NetworkMix(channels,self.metadata, layers, next_arch_ops, next_arch_kernel)             
                   
                    next_arch_train_acc, next_arch_test_acc  = self.train(epochs,model)

                    if next_arch_test_acc > archbestv:
                        archbestt = next_arch_train_acc
                        archbestv = next_arch_test_acc
                        model = self.model

                    logging.info("Candidate Train Acc %f Candidate Val Acc %f", next_arch_train_acc, next_arch_test_acc)
            
            
            
                next_arch_train_acc = archbestt
                next_arch_test_acc = archbestv
                logging.info("Candidate Best Train %f Candidate Best Val %f", next_arch_train_acc, next_arch_test_acc)

                candidate_count += 1


                if (next_arch_test_acc > curr_arch_test_acc + 0.40):
                    self.save_checkpoint(model, self.best_epoch)

                    # update current architecture.
                    curr_arch_ops = next_arch_ops
                    curr_arch_kernel = next_arch_kernel
                    f_layers = len(curr_arch_ops)
                    f_channels = channels
                    f_epochs = epochs

                    logging.info("Candidate Train Acc %f Candidate Val Acc %f", next_arch_train_acc, next_arch_test_acc)
                    logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)                        
                    logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
                    
                    curr_arch_train_acc = next_arch_train_acc                
                    curr_arch_test_acc = next_arch_test_acc

                    bst_dep_2.append(layers)
                    bst_wdt_2.append(channels)
                    bst_epc_2.append(epochs)
                    bst_tac_2.append(round(next_arch_train_acc,2))
                    bst_vac_2.append(round(next_arch_test_acc,2))
                    bst_prm_2.append(general_num_params(model))

                    logging.info('Best Arch Layers: %s', bst_dep_2)
                    logging.info('Best Arch Channels: %s', bst_wdt_2)
                    logging.info('Best Arch Epochs: %s', bst_epc_2)
                    logging.info('Best Arch Train Acc: %s', bst_tac_2)
                    logging.info('Best Arch Val Acc: %s', bst_vac_2)
                    logging.info('Best Arch Params: %s', bst_prm_2)

                else:
                    logging.info("Candidate Train Acc %f Candidate Val Acc %f", next_arch_train_acc, next_arch_test_acc)
                    logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)                        
                    logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
                logging.info('#############################################################################')

            bst_dep = bst_dep_2
            bst_wdt = bst_wdt_2
            bst_epc = bst_epc_2
            bst_tac = bst_tac_2
            bst_vac = bst_vac_2
            bst_prm = bst_prm_2

            bst_dep,bst_wdt,bst_epc,bst_tac,bst_vac,bst_prm = self.sort_networks(bst_dep,bst_wdt,bst_epc,bst_tac,bst_vac,bst_prm)


            logging.info('Best Arch Layers: %s', bst_dep)
            logging.info('Best Arch Channels: %s', bst_wdt)
            logging.info('Best Arch Epochs: %s', bst_epc)
            logging.info('Best Arch Train Acc: %s', bst_tac)
            logging.info('Best Arch Val Acc: %s', bst_vac)
            logging.info('Best Arch Params: %s', bst_prm)
            
            ###################################################################################
            ###################################################################################
            ########### HYPER PARAMETER SEARCH ENDS ############################
        logging.info('Discovered Final Depth %s', f_layers)
        logging.info('Discovered Final Width %s', f_channels)
        logging.info('Discovered Final Epochs %s', f_epochs)



        return curr_arch_ops, curr_arch_kernel, f_channels, f_layers

    """
    ====================================================================================================================
    SEARCH =============================================================================================================
    ====================================================================================================================
    The search function is called with no arguments, and expects a PyTorch model as output. Use this to
    perform your architecture search. 

    """


    def evaluate(self):
        self.model.eval()
        labels, predictions = [], []
        for data, target in self.valid_loader:
            #data = data.to(self.device)
            data = data.cuda(non_blocking=True)
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

    def search(self):
       
        curr_arch_ops, curr_arch_kernel, f_channels, f_layers = self.search_depth_and_width()
        # curr_arch_ops = [0, 0, 0 ,0, 0 ,0, 0]
        # curr_arch_kernel = [3, 3, 3, 3, 3, 3, 3]
        # f_channels = 16
        # f_layers = 7
        model = NetworkMix(f_channels, self.metadata, f_layers, curr_arch_ops, curr_arch_kernel)
        return model


    def sort_networks(self, d, w, e, t, v, p):

        if len(d)==0:
            return d,w,e,t,v,p
        # Sorting all lists based on the sorted order of list1
        combined = list(zip(d, w, e, t, v, p))

        # Sort based on the first list (list1)
        combined_sorted = sorted(combined, key=lambda x: x[5])

        # Unzip the sorted combined list to separate the lists
        d, w, e, t, v, p = zip(*combined_sorted)

        # Convert them back to lists (since zip() returns tuples)
        d = list(d)
        w = list(w)
        e = list(e)
        t = list(t)
        v = list(v)
        p = list(p)

        # Print the sorted lists
        print("Sorted depth:", d)
        print("Sorted width:", w)
        print("Sorted epochs:", e)

        print("Sorted tac:", t)
        print("Sorted vac:", v)
        print("Sorted params:", p)

        return d,w,e,t,v,p
