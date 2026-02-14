import torchvision
import torch.nn as nn
import torch
import torch

from torch import optim
from helpers import NetworkMix , show_time, general_num_params, Clock, set_seed, log_lines
import logging, sys, os
from sklearn.metrics import accuracy_score
import numpy as np
import time
import copy
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from collections import Counter


import subprocess
import json

def get_gpu_memory():
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader']
    )
    used, total = map(int, output.decode().strip().split(','))
    return used, total

def assert_memory_limit(max_mb=2550):
    used, total = get_gpu_memory()
    if used > max_mb:
        # Raise with 'out of memory' in the message so it is caught in NAS.train
        raise RuntimeError(f"CUDA out of memory: GPU memory usage exceeded {max_mb} MB (used: {used} MB)")
    



log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('Unseen1.txt'))
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
    #search_time_limit = 5 * 60  # Hardcoded: 4 hours in seconds for NAS search (change as needed)
    
    def __init__(self, train_loader, valid_loader, metadata, clock):

        log_lines(2)

    
        search_size = 0.99
        
        data = train_loader.dataset.x  # The data samples
        labels = train_loader.dataset.y
        
        train_indices, val_indices = train_test_split(range(len(labels)),
                                                      test_size=1-search_size,  # 25% of the data
                                                      stratify=labels#,  # Preserve class distribution
                                                     # random_state=42  # For reproducibility
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

        # Removed debug print statements
        ########################################################################################################
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

        # === Time Budget Logic (using helpers.get_per_dataset_time_limits) ===
        from helpers import get_per_dataset_time_limits
        # Use the time_limit_in_hours argument passed to Clock in main.py
        if not hasattr(NAS, '_total_time'):
            # Fallback: try to get from clock.time_limit and clock.start_time
            if hasattr(clock, 'time_limit') and hasattr(clock, 'start_time'):
                NAS._total_time = float(clock.time_limit - clock.start_time)
            else:
                NAS._total_time = 15 * 60  # fallback: 15 minutes in seconds
                
        n_datasets = int(metadata.get('n_datasets', 3))
        dataset_idx = int(metadata.get('dataset_idx', NAS._dataset_counter-1 if hasattr(NAS, '_dataset_counter') else 0))
        
        total_time = NAS._total_time
        # Default: 50% for search, 50% for train (can be changed)
        search_time, train_time, extra_time = get_per_dataset_time_limits(total_time, dataset_idx, n_datasets, search_ratio=0.5, train_ratio=0.5)
        self.phase1_time_out = False
        self.phase2_time_out = False
        self.search_time_limit = search_time
        self.train_time_limit = train_time
        self.extra_time = extra_time
        self.metadata['train_time_limit'] = self.train_time_limit
        logging.info(f"[NAS] Dataset {dataset_idx+1}/{n_datasets}: Allocated {show_time(search_time+train_time)} (search: {show_time(search_time)}, train: {show_time(train_time)}, extra: {show_time(extra_time)})")
        
        self.total_time_limit = getattr(self, 'search_time_limit', None)
        self.phase1_time_limit = self.total_time_limit * 0.40 if self.total_time_limit else None
        self.phase1_time_limit -= self.extra_time
        self.phase2_time_limit = self.total_time_limit * 0.60 if self.total_time_limit else None
        self.phase2_time_limit -= self.extra_time


        
        logging.info(f"Total Search Time: {self.total_time_limit})")
        logging.info(f"Phase 1 Search Time: {self.phase1_time_limit})")
        logging.info(f"Phase 2 Search Time: {self.phase2_time_limit})")
        
    

    
    def train(self, epochs, model, phase_check):
    
        self.model = model.to(self.device)  # Always move model to the selected device (CPU or GPU)
        self.epochs = epochs  # Set number of epochs
        self.optimizer = optim.SGD(self.model.parameters(), lr=.01, momentum=.9, weight_decay=3e-4)  
        self.criterion = nn.CrossEntropyLoss()  # Set loss function
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs*20) 

        # Log device info for worst-case runtime estimation
        if self.device.type == 'cuda':
            logging.info('Running on GPU (CUDA)')
        else:
            logging.info('Running on CPU (worst-case scenario)')

        t_start = time.time()
        best_valid_acc = 0.0  # Track best validation accuracy
        best_train_acc = 0.0  # Track best training accuracy
        best_epoch = 0  # Track best epoch
        self.best_model = None  # Track best model
        batch_size = self.train_loader.batch_size if hasattr(self.train_loader, 'batch_size') else 16  # Get initial batch size
       
        try:
            
            for epoch in range(epochs):
                #check time
                if phase_check == 'phase1':
                    if self.phase1_time_limit is not None and (time.time() - self.search_start) > self.phase1_time_limit:
                        logging.info(f"Phase 1 (main search) time limit of {self.phase1_time_limit/60:.2f} min exceeded. Moving to phase 2.")
                        self.phase1_time_out = True
                        break
                elif phase_check == 'phase2':
                    if self.phase2_time_limit is not None and (time.time() - self.search_end) > self.phase2_time_limit:
                        logging.info(f"Phase 2 (hyperparameter search) time limit of {self.phase2_time_limit/60:.2f} min exceeded. Stopping search.")
                        self.phase2_time_out = True
                        break
                else:
                    pass
                self.model.train()
                labels, predictions = [], []
                for data, target in self.train_loader:
                    data, target = data.to(self.device), target.to(self.device)  # Always move data to the selected device
                    self.optimizer.zero_grad()
                    output = self.model.forward(data)
                    labels += target.cpu().tolist()  # Store labels
                    predictions += torch.argmax(output, 1).detach().cpu().tolist()  # Store predictions
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                # cuda error check    
                    #assert_memory_limit(max_mb=4096)  # Adjust as needed, e.g., 4096 MB for 4GB GPU

                
                train_acc = accuracy_score(labels, predictions)  # Compute training accuracy
                valid_acc = self.evaluate()  # Compute validation accuracy
                

                
                logging.info("\tEpoch {:>3}/{:<3} | Train Acc: {:>6.2f}% | Valid Acc: {:>6.2f}% | T/Epoch: {:<7} | LR: {:>2.6f} |".format(
                epoch + 1, self.epochs,
                train_acc * 100, valid_acc * 100,
                show_time((time.time() - t_start) / (epoch + 1)),
                self.scheduler.get_last_lr()[0]
                ))                    
                #self.scheduler.step()  # Step the scheduler                     
                # Update best model if validation accuracy improves
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    best_train_acc = train_acc
                    self.best_epoch = epoch + 1
                    self.best_model = copy.deepcopy(self.model)

        except RuntimeError as e:
            # Handle CUDA OOM by reducing batch size and retrying
            if 'CUDA out of memory' in str(e) or 'memory' in str(e):
                logging.warning(f"CUDA OOM at batch size {batch_size}.")
                torch.cuda.empty_cache()
                return 0.0, 0.0  # Return 0 accuracies if OOM occurs
            else:
                raise
        #self.model = best_model  # Restore best model
        logging.info("Candidate Evaluation Time: {}".format(show_time(time.time() - t_start)))
        return best_train_acc * 100, best_valid_acc * 100  # Return best accuracies as percentages


    def get_candidate_model(self, layers, channels):


        # Returns a model Baseline Model 
        curr_arch_ops = next_arch_ops = np.zeros((layers,), dtype=int)
        #curr_arch_ops = next_arch_ops = np.ones((layers,), dtype=int)

        curr_arch_kernel = next_arch_kernel = 3*np.ones((layers,), dtype=int)


        model = NetworkMix(channels,self.metadata, layers, curr_arch_ops, curr_arch_kernel)          
        logging.info("Model Parameters = %f", general_num_params(model))

        return model

    def search_depth_and_width(self):
        self.search_start = time.time()  # Track search start time for time limit enforcement
        
        logging.info('RUNNING SEARCH on %s', self.metadata['codename'])

        
        min_dim = min(self.metadata['input_shape'][2],self.metadata['input_shape'][3])
        max_dim = max(self.metadata['input_shape'][2],self.metadata['input_shape'][3])
        data_channels = self.metadata['input_shape'][1]

        total_input_pts = self.metadata['input_shape'][1]*self.metadata['input_shape'][2]*self.metadata['input_shape'][3]

        if min_dim >= 96: #if input > 3x64x64
            max_params = 3_500_000
        elif min_dim >= 48:
            max_params = 3_000_000
        elif min_dim >=  24:
            max_params = 2_500_000
        elif min_dim >= 12:
            max_params = 1_500_000    
        else:
            max_params = 1_000_000   


        target_acc= 100
        min_width=  16
        max_width= 2048
        depth_resolution = 4
        min_depth= 8
        max_depth= 100
        max_epochs =50
        Rand_train = 5
        max_models = 9
        
        r1_thresh = 0.05
        r2_thresh = 0.10

        channels = f_channels = min_width#16 
        layers = min_depth
        
        add_epochs = 1# 1
        s_epoch = epochs = 2#1
        f_epochs = 0

        macro_count = 0

        bst_dep = []
        bst_wdt = []
        bst_epc = []
        bst_tac = []
        bst_vac = []
        bst_prm = []


        # Train Baseline Model
        curr_arch_train_acc = next_arch_train_acc = 0.0
        curr_arch_test_acc = next_arch_test_acc = 0.0
        logging.info('Evaluating Baseline Model...')
        model = self.get_candidate_model(layers, channels)
        logging.info("Model Depth %s Model Width %s Train Epochs %s", layers, channels, epochs)

        curr_arch_train_acc, curr_arch_test_acc  = self.train(epochs, model, 'baseline')
        self.save_checkpoint(self.best_model, self.epochs) 
        logging.info("Baseline Train Acc %f Baseline Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)


        # Create equally spaced models with respect to parameters
        arr = np.linspace(general_num_params(model), max_params, max_models, dtype=int)
        candidate_layers = []
        candidate_channels = []
        candidate_params = []

        for i in range(1,len(arr)):            
            #logging.info("Required Depth and Width for Model Parameters = %s", arr[i])
            ch_up = True
            ly_up = True

            while(general_num_params(model) < arr[i]):
                if ch_up:
                    channels+= 8
                    ch_up=False
                else:
                    layers+=1

                    ch_up=True


                curr_arch_ops = next_arch_ops = np.zeros((layers,), dtype=int)
                curr_arch_kernel = next_arch_kernel = 3*np.ones((layers,), dtype=int)
                model = NetworkMix(channels,self.metadata, layers, curr_arch_ops, curr_arch_kernel)        
                last_model_params = general_num_params(model)

                #logging.info("Model Depth %s Model Width %s Model Parameters %s", layers, channels, last_model_params)
                
            candidate_layers.append(layers)
            candidate_channels.append(channels)    
            candidate_params.append(last_model_params)
   
        logging.info('Required Parameters Layers: %s', arr[1:])
        logging.info('Candidate Layers: %s', candidate_layers)
        logging.info('Candidate Channels: %s', candidate_channels)
        logging.info('Candidate Params: %s', candidate_params)
        log_lines(2)


        
        # Initiate Search Phase 1
        self.best_model_rand = self.best_model
        self.best_epoch_rand = self.epochs
        # SEARCH MODEL
        epochs_up = False
        candidate_count = 0 # Independent Repeat Model Counter
        candidate_num = 0 # Loop Iterator

        while (curr_arch_test_acc < target_acc):
            
            torch.cuda.empty_cache()
            if self.phase1_time_out:
                self.phase1_time_out = False
                break
            
            if (curr_arch_train_acc >= 99):
                break;  
                                
            if epochs_up and epochs < max_epochs:                                
                epochs = epochs + add_epochs
                epochs_up = False

            if candidate_num < len(candidate_layers):
                layers = candidate_layers[candidate_num]
                channels = candidate_channels[candidate_num]
            else:
                break

            log_lines(2)
            logging.info('Moving to Next Candidate Architecture...')
            logging.info("Model Depth %s Model Width %s Train Epochs %s", layers, channels, epochs)

            archbestt = curr_arch_train_acc
            archbestv = curr_arch_test_acc            

            # Training same candidate with multiple initializations.    
            for i in range(Rand_train):
                torch.cuda.empty_cache()                    
                set_seed(i)
                logging.info("INITIALIZING RUNNUNG RUN %f", i) 
                model = self.get_candidate_model(layers, channels)             
                next_arch_train_acc, next_arch_test_acc  = self.train(epochs,model,'phase1')

                if next_arch_test_acc > archbestv + r1_thresh:
                    archbestt = next_arch_train_acc
                    archbestv = next_arch_test_acc
                    self.best_model_rand = self.best_model
                    self.best_epoch_rand = self.epochs
                    break                  
                logging.info("Candidate Train Acc %f Candidate Val Acc %f", next_arch_train_acc, next_arch_test_acc)
                log_lines(1) 


            # As long as we get significant improvement by increasing depth.
            
            #next_arch_train_acc = archbestt
            #next_arch_test_acc = archbestv
            #logging.info("Candidate Best Train %f Candidate Best Val %f", next_arch_train_acc, next_arch_test_acc)

            if (next_arch_test_acc > curr_arch_test_acc + r1_thresh):


                curr_arch_ops = next_arch_ops
                curr_arch_kernel = next_arch_kernel
                logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)
                curr_arch_train_acc = next_arch_train_acc
                curr_arch_test_acc = next_arch_test_acc
                f_channels = channels
                f_epochs = epochs
                s_epoch = self.best_epoch
                self.save_checkpoint(self.best_model_rand, self.best_epoch_rand)
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

                candidate_num += 1
                candidate_count = 0


            else:
                candidate_count += 1

                logging.info('Current Candidate Evaluation Count: %f',candidate_count) 
                logging.info("Highest Train Acc %f Highest Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)                        
                logging.info("Train Acc Diff %f Val Acc Diff %f", next_arch_train_acc-curr_arch_train_acc, next_arch_test_acc-curr_arch_test_acc)

                if candidate_count < 3:
                    epochs_up=True
                else:
                    candidate_num += 1   
                    candidate_count = 0 

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
        
        Rand_train = 3
        candidate_count = 0
        # Phase 2: Hyperparameter search/refinement
        
        self.search_end = time.time()  # Track search start time for time limit enforcement
        
        phase1_time = self.search_end - self.search_start
        logging.info(f"Phase 1 Search Time: {phase1_time})")
        


        
        self.phase2_time_limit = self.total_time_limit - phase1_time
        self.phase2_time_limit -= self.extra_time

        logging.info(f"Phase 2 Allocated Time: {self.phase2_time_limit})")
        

        
        
        iter_phase2 = 0
        while len(bst_dep) > 1:
            
            iter_phase2 +=1
            # Check if phase 2 time limit is exceeded
            if self.phase2_time_out:
                self.phase2_time_out = False
                break
            bst_dep_2 = []
            bst_wdt_2 = []
            bst_epc_2 = []
            bst_tac_2 = []
            bst_vac_2 = []
            bst_prm_2 = []

            #if candidate_count > max_models:
            #    break
            for i in range(len(bst_dep)):
                layers = bst_dep[len(bst_dep)-2-i]
                channels = bst_wdt[len(bst_wdt)-2-i]

                if layers <= 8:
                    pass

                if i == 0:
                    epochs = max(f_epochs,bst_epc[len(bst_epc)-1]) + 1
                else:
                    epochs = bst_epc[len(bst_epc)-1] + i + 1
                    #epochs = epochs + iter_phase2
                
                archbestt, archbestv = 0.0, 0.0                

                next_arch_ops = np.zeros((layers,), dtype=int)
                #next_arch_ops = np.ones((layers,), dtype=int)

                next_arch_kernel = 3*np.ones((layers,), dtype=int)
                model = NetworkMix(channels,self.metadata, layers, next_arch_ops, next_arch_kernel)                             
                
                logging.info('Moving to Next Candidate Architecture...')
                logging.info("Model Depth %s Model Width %s Train Epochs %s", layers, channels, epochs)
                logging.info("Model Parameters = %f", general_num_params(model))
                #log_networkmix_layer_outputs(model, self.metadata)



                for i in range(Rand_train):
                    torch.cuda.empty_cache()
                    if self.phase2_time_limit is not None and (time.time() - self.search_end) > self.phase2_time_limit:
                        logging.info(f"Phase 2 (hyperparameter search) time limit of {self.phase2_time_limit/60:.2f} min exceeded. Stopping search.")
                        break
                    set_seed(i)


                    logging.info("INITIALIZING RUNNUNG RUN %f", i) 
                    model = NetworkMix(channels,self.metadata, layers, next_arch_ops, next_arch_kernel)             

                    next_arch_train_acc, next_arch_test_acc  = self.train(epochs,model,'phase2')

                    if next_arch_test_acc > archbestv:
                        archbestt = next_arch_train_acc
                        archbestv = next_arch_test_acc
                        self.best_model_rand = self.best_model
                        self.best_epoch_rand = self.epochs
                    logging.info("Candidate Train Acc %f Candidate Val Acc %f", next_arch_train_acc, next_arch_test_acc)
            
            
            
                next_arch_train_acc = archbestt
                next_arch_test_acc = archbestv
                logging.info("Candidate Best Train %f Candidate Best Val %f", next_arch_train_acc, next_arch_test_acc)

                candidate_count += 1


                if (next_arch_test_acc > curr_arch_test_acc + r2_thresh):
                    self.save_checkpoint(self.best_model_rand, self.best_epoch_rand)

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

        log_lines(10)
        
        phase2_time_taken = time.time() - self.search_end
        logging.info(f"Phase 2 Taken Time: {phase2_time_taken})")

        return curr_arch_ops, curr_arch_kernel, f_channels, f_layers
        
        '''
        '''
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
            'architecture':model,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.metadata["codename"]+".pth")
        print(f"Checkpoint saved to {self.metadata['codename']}.pth")

    def search(self):
       
        curr_arch_ops, curr_arch_kernel, f_channels, f_layers = self.search_depth_and_width()
        #self.search_depth_and_width()
        #curr_arch_ops = [0, 0, 0 ,0, 0 ,0, 0]
        #curr_arch_kernel = [3, 3, 3, 3, 3, 3, 3]
        #f_channels = 16
        #f_layers = 7
        model = NetworkMix(f_channels, self.metadata, f_layers, curr_arch_ops, curr_arch_kernel)
        log_networkmix_layer_outputs(model, self.metadata)
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

        return d,w,e,t,v,p

def log_networkmix_layer_outputs(model, metadata=None):
    # Log the output shapes of each layer using torch tensors instead of torchinfo.summary
    try:
        model.eval()
        if metadata is not None and 'input_shape' in metadata:
            input_shape = metadata['input_shape'][1:]  # skip batch dim
        else:
            input_shape = (model.stem[0].in_channels, 128, 128)  # fallback
        dummy = torch.zeros((1, *input_shape)).to(next(model.parameters()).device)
        logging.info(f"Input shape: {dummy.shape}")
        hooks = []
        layer_outputs = {}
        def hook_fn(name):
            def fn(module, inp, out):
                layer_outputs[name] = out.shape if hasattr(out, 'shape') else str(type(out))
            return fn
        for name, module in model.named_modules():
            if name == '':
                continue  # skip the root module
            hooks.append(module.register_forward_hook(hook_fn(name)))
        with torch.no_grad():
            _ = model(dummy)
        for name, shape in layer_outputs.items():
            logging.info(f"Layer: {name}, Output shape: {shape}")
        for h in hooks:
            h.remove()
    except Exception as e:
        logging.info(f" Failed to log layer outputs: {e}")
    # torchinfo.summary is commented out
    # try:
    #     from torchinfo import summary
    #     if metadata is not None and 'input_shape' in metadata:
    #         input_shape = metadata['input_shape'][1:]  # skip batch dim
    #     else:
    #         input_shape = (model.stem[0].in_channels, 128, 128)  # fallback
    #     logging.info(str(summary(model, input_size=(1, *input_shape), depth=3)))  # Print model summary
    # except Exception:
    #     logging.info("torchinfo.summary not available or failed.")
    
'''
'''