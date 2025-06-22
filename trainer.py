import time
from sklearn.metrics import accuracy_score

import torch
from torch import optim
import torch.nn as nn
from helpers import show_time
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
    def __init__(self, model, device, train_dataloader, valid_dataloader, lr, epochs, metadata):
        self.model = model
        self.device = device
        #self.model = model.to(self.device)
        self.model = model.cuda()

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.metadata = metadata

        # define  training parameters
        self.lr = lr
        self.epochs = epochs
        self.epc_mult = 50        
        self.optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=.9, weight_decay=3e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        #checkpoint = torch.load(self.metadata["codename"]+".pth", map_location=self.device)
        #self.model.load_state_dict(checkpoint['model_state_dict'])
        #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #self.start_epochs = checkpoint['epoch']
        #print(self.start_epochs)
        #self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.start_epochs*self.epc_mult)
        #self.epochs = self.start_epochs*self.epc_mult
    """
    ====================================================================================================================
    TRAIN ==============================================================================================================
    ====================================================================================================================
    The train function will define how your model is trained on the train_dataloader.
    Output: Your *fully trained* model
    
    See the example submission for how this should look
    """

    def train(self):

        final_train_time_secs = 3600*4
        t_start = time.time()

        best_model = None
        best_valid_acc = 0.0

        #for epoch in range(self.start_epochs, self.epochs):
        for epoch in range(0, self.epochs):
            self.model.train()
            labels, predictions = [], []

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

            logging.info("\tEpoch {:>3}/{:<3} | Train Acc: {:>6.2f}% | Valid Acc: {:>6.2f}% | T/Epoch: {:<7} | LR: {:>2.6f} |".format(
                epoch + 1, self.epochs,
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


            if time.time() - t_start > final_train_time_secs:
                logging.info("Final Training Time Exceeded Given Limit")
                break

        logging.info("Total final training runtime: {}".format(show_time(time.time() - t_start)))
        self.model = best_model
         # Log the best model stats after training ends
        logging.info("Best Model Stats: Epoch {:>3}, Train Acc: {:>6.2f}%, Valid Acc: {:>6.2f}%".format(
        best_epoch, best_train_acc * 100, best_valid_acc * 100
        ))

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
