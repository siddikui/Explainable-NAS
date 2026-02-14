import time
from sklearn.metrics import accuracy_score

import torch
from torch import optim
import torch.nn as nn
from helpers import show_time, get_per_dataset_time_limits, log_lines, set_seed
import logging
import copy


class Trainer:
    """
    ====================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The Trainer class will receive the following inputs
        * model: The model returned by your NAS class
        * train_loader: The train loader created by your DataProcessor
        * valid_loader: The valid loader created by your DataProcessor
        * metadata: A dictionary with information about this dataset, with the following keys:
            'num_classes' : The number of output classes in the classification problem
            'codename' : A unique string that represents this dataset
            'input_shape': A tuple describing [n_total_datapoints, channel, height, width] of the input data
            'time_remaining': The amount of compute time left for your submission
            plus anything else you added in the DataProcessor or NAS classes
    """
    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata, clock):
        self.model = model
        self.device = device
        self.model = model.cuda()
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.metadata = metadata
        self.clock = clock
        # Set default training parameters
        self.lr = 0.01
        self.search_epochs = [600]

        self.epc_mult = 20        
        self.optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=.9, weight_decay=3e-4)
        self.criterion = nn.CrossEntropyLoss()


        # === Time Budget Logic (using helpers.get_per_dataset_time_limits) ===
        from helpers import get_per_dataset_time_limits
        # Use the time_limit_in_hours argument passed to Clock in main.py
        if not hasattr(Trainer, '_total_time'):
            # Fallback: try to get from clock.time_limit and clock.start_time
            if hasattr(clock, 'time_limit') and hasattr(clock, 'start_time'):
                Trainer._total_time = float(clock.time_limit - clock.start_time)
            else:
                Trainer._total_time = 15 * 60  # fallback: 15 minutes in seconds
                
        n_datasets = int(metadata.get('n_datasets', 3))
        dataset_idx = int(metadata.get('dataset_idx', 0))
        
        total_time = Trainer._total_time
        
        search_time, train_time, extra_time = get_per_dataset_time_limits(total_time, dataset_idx, n_datasets, search_ratio=0.5, train_ratio=0.5)
        
        self.train_time_limit = train_time
        self.extra_time = extra_time
        
        logging.info(f"[Trainer] Dataset {dataset_idx+1}/{n_datasets}: Time allocation - Search: {show_time(search_time)}, Train: {show_time(train_time)}, Extra: {show_time(extra_time)}.")

        '''
        '''
        self.checkpoint = torch.load(self.metadata["codename"]+".pth", map_location=self.device)
        #self.model.load_state_dict(self.checkpoint['model_state_dict'])
        #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epochs = self.checkpoint['epoch']
        #print(self.start_epochs)
        #self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.epochs = self.start_epochs*self.epc_mult
        
    """
    ====================================================================================================================
    TRAIN ==============================================================================================================
    ====================================================================================================================
    The train function will define how your model is trained on the train_dataloader.
    Output: Your *fully trained* model
    
    See the example submission for how this should look
    """
    def epoch_search(self, t_start):
        Rand_train = 1
        max_epochs = 20
        #for epoch in range(self.start_epochs, self.epochs):
        self.checkpoint = torch.load(self.metadata["codename"]+".pth", map_location=self.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        best_valid_acc = self.evaluate()

        logging.info("Best Searched Validation Accuracy: {:>6.2f}%".format(best_valid_acc * 100))

        for s_epochs in self.search_epochs:
            # t_start = time.time()
            
            for i in range(Rand_train):
                torch.cuda.empty_cache()
                    
                set_seed(i)
                self.checkpoint = torch.load(self.metadata["codename"]+".pth", map_location=self.device)
                self.model.load_state_dict(self.checkpoint['model_state_dict'])
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=.9, weight_decay=3e-4)
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=s_epochs)

                max_epochs = s_epochs
                for epoch in range(s_epochs):
                    if epoch > max_epochs:
                        logging.info("Early stopping at epoch 20")
                        break
                    self.model.train()
                    labels, predictions = [], []

                                # Enforce the per-dataset training time limit
                    if time.time() - t_start > self.train_time_limit-self.extra_time:
                        logging.info("Hyperparamter Training Time Exceeded Given Limit")
                        break

                    for data, target in self.train_dataloader:
                        #data, target = data.to(self.device), target.to(self.device)
                        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
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

                    logging.info("EPOCH_LOG | Epoch {:03d}/{} | Train Acc: {:6.2f}% | Valid Acc: {:6.2f}% | Time/Epoch: {:<7} | LR: {:.6f}".format(
                        epoch + 1, s_epochs,
                        train_acc * 100, valid_acc * 100,
                        show_time((time.time() - t_start) / (epoch + 1)),
                        self.scheduler.get_last_lr()[0]
                    ))

                    self.scheduler.step()


                    # Check if this is the best model based on validation accuracy
                    if valid_acc > best_valid_acc:
                        best_valid_acc = valid_acc
                        best_train_acc = train_acc
                        best_epoch = epoch + 1
                        best_search_epoch = s_epochs
                        # best_model = copy.deepcopy(self.model)  # Keep a copy of the best model
                        # Log model architecture when a new best model is found


                    # Enforce the per-dataset training time limit
                    if time.time() - t_start > self.train_time_limit-self.extra_time:
                        logging.info("Final Training Time Exceeded Given Limit")
                        break
                logging.info("Best Model Stats: Epoch {:>3}, Train Acc: {:>6.2f}%, Valid Acc: {:>6.2f}%".format(
                best_epoch, best_train_acc * 100, best_valid_acc * 100
                ))

        return best_search_epoch

    def train(self):
        t_start = time.time()
        best_epoch = 100  
        best_model = self.model
        best_valid_acc = 0.0
        best_train_acc = 0.0
        epochs = 800#self.epoch_search(t_start)#self.epochs#

        self.checkpoint = torch.load(self.metadata["codename"]+".pth", map_location=self.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=.9, weight_decay=3e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        best_valid_acc = self.evaluate()

        logging.info("Best Searched Validation Accuracy: {:>6.2f}%".format(best_valid_acc * 100))
                
        for epoch in range(epochs):

            self.model.train()
            labels, predictions = [], []

                        # Enforce the per-dataset training time limit
            if time.time() - t_start > self.train_time_limit-self.extra_time:
                logging.info("Hyperparamter Training Time Exceeded Given Limit")
                break

            for data, target in self.train_dataloader:
                #data, target = data.to(self.device), target.to(self.device)
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
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

            logging.info("EPOCH_LOG | Epoch {:03d}/{} | Train Acc: {:6.2f}% | Valid Acc: {:6.2f}% | Time/Epoch: {:<7} | LR: {:.6f}".format(
                epoch + 1, epochs,
                train_acc * 100, valid_acc * 100,
                show_time((time.time() - t_start) / (epoch + 1)),
                self.scheduler.get_last_lr()[0]
            ))

            self.scheduler.step()


            # Check if this is the best model based on validation accuracy
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_train_acc = train_acc
                best_epoch = epoch + 1
                best_model = copy.deepcopy(self.model)  # Keep a copy of the best model
                # Log model architecture when a new best model is found


            # Enforce the per-dataset training time limit
            if time.time() - t_start > self.train_time_limit-self.extra_time:
                logging.info("Final Training Time Exceeded Given Limit")
                break

        logging.info("Total final training runtime: {}".format(show_time(time.time() - t_start)))
        self.model = best_model
         # Log the best model stats after training ends
        logging.info("Best Model Stats: Epoch {:>3}, Train Acc: {:>6.2f}%, Valid Acc: {:>6.2f}%".format(
        best_epoch, best_train_acc * 100, best_valid_acc * 100
        ))
        log_lines(20)
        return self.model  # Return the model with the best validation accuracy

    # print out the model's accuracy over the valid dataset
    # (this isn't necessary for a submission, but I like it for my training logs)
    def evaluate(self):
        self.model.eval()
        labels, predictions = [], []
        for data, target in self.valid_dataloader:
            #data = data.to(self.device)
            data = data.cuda(non_blocking=True)

            output = self.model.forward(data)
            labels += target.cpu().tolist()
            predictions += torch.argmax(output, 1).detach().cpu().tolist()
        return accuracy_score(labels, predictions)


    """
    ====================================================================================================================
    PREDICT ============================================================================================================
    ====================================================================================================================
    The prediction function will define how the test dataloader will be passed through your model. It will receive:
        * test_dataloader created by your DataProcessor
    
    And expects as output:
        A list/array of predicted class labels of length=n_test_datapoints, i.e, something like [0, 0, 1, 5, ..., 9] 
    
    See the example submission for how this should look.
    """

    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        for data in test_loader:
            #data = data.to(self.device)
            data = data.cuda(non_blocking=True)

            output = self.model.forward(data)
            predictions += torch.argmax(output, 1).detach().cpu().tolist()
        return predictions
