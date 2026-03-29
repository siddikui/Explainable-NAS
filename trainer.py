import time
from sklearn.metrics import accuracy_score

import torch
from torch import optim
import torch.nn as nn
from helpers import show_time, get_per_dataset_time_limits, log_lines
import logging
import copy

class Trainer:
 
    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata, clock):
        self.model = model
        self.device = device
        self.model = model.cuda()
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.metadata = metadata
        self.clock = clock
 
        self.lr = 0.025                  # CHANGED: 0.01 -> 0.025 (DARTS eval setting)
        self.epochs = 600
        self.epc_mult = 50
 
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=3e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )
 
        # ADDED: auxiliary loss weight — same as DARTS eval
        # NetworkCIFAR returns (logits, logits_aux) when auxiliary=True
        # and training=True. The aux loss is weighted at 0.4 and added
        # to the main loss to provide extra gradient signal.
        self.auxiliary_weight = 0.4
 
        # ADDED: drop path prob — linearly increases during training
        # Set on model externally before each forward pass.
        # Helps regularize deep networks during final training.
        self.drop_path_prob = 0.2
 
        from helpers import get_per_dataset_time_limits
        if not hasattr(Trainer, '_total_time'):
            if hasattr(clock, 'time_limit') and hasattr(clock, 'start_time'):
                Trainer._total_time = float(clock.time_limit - clock.start_time)
            else:
                Trainer._total_time = 15 * 60
 
        n_datasets  = int(metadata.get('n_datasets', 3))
        dataset_idx = int(metadata.get('dataset_idx', 0))
        total_time  = Trainer._total_time
 
        search_time, train_time, extra_time = get_per_dataset_time_limits(
            total_time, dataset_idx, n_datasets, search_ratio=0.5, train_ratio=0.5)
 
        self.train_time_limit = train_time
        self.extra_time       = extra_time
 
        logging.info(f"[Trainer] Dataset {dataset_idx+1}/{n_datasets}: Time allocation - "
                     f"Search: {show_time(search_time)}, Train: {show_time(train_time)}, "
                     f"Extra: {show_time(extra_time)}.")
 
        # REMOVED: the checkpoint load block that was here.
        # It was trying to load a Network (search) state_dict into
        # NetworkCIFAR (eval) — completely different architectures,
        # hence all the Missing/Unexpected key errors.
        # The model passed in from search() is already built correctly
        # from the genotype — just train it from scratch, no loading needed.
 
 
    def train(self):
        t_start = time.time()
 
        best_model     = None
        best_valid_acc = 0.0
        best_train_acc = 0.0
        best_epoch     = 0
 
        for epoch in range(0, self.epochs):
 
            if time.time() - t_start > self.train_time_limit - self.extra_time:
                logging.info("Final Training Time Exceeded Given Limit")
                break
 
            self.model.train()
 
            # ADDED: set drop_path_prob linearly increasing each epoch.
            # DARTS does this: starts at 0, reaches max at final epoch.
            # Must be set on model before forward pass each epoch.
            self.model.drop_path_prob = self.drop_path_prob * epoch / self.epochs
 
            labels, predictions = [], []
 
            for data, target in self.train_dataloader:
                data   = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                target = target.to(torch.int64)
 
                self.optimizer.zero_grad()
 
                # CHANGED: NetworkCIFAR returns (logits, logits_aux) — a tuple.
                # Your original did output = model(data) then argmax(output)
                # which crashes because output is a tuple not a tensor.
                # Unpack both values here.
                logits, logits_aux = self.model.forward(data)
 
                loss = self.criterion(logits, target)
 
                # ADDED: auxiliary loss — only during training, only if
                # auxiliary=True was set when building NetworkCIFAR.
                # logits_aux is None when auxiliary=False or model.eval().
                if logits_aux is not None:
                    loss_aux = self.criterion(logits_aux, target)
                    loss = loss + self.auxiliary_weight * loss_aux
 
                labels      += target.cpu().tolist()
                predictions += torch.argmax(logits, 1).detach().cpu().tolist()
 
                loss.backward()
 
                # UNCHANGED: grad clipping already in your trainer — good.
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()
 
            train_acc = accuracy_score(labels, predictions)
            valid_acc = self.evaluate()
 
            logging.info("\tEpoch {:>3}/{:<3} | Train Acc: {:>6.2f}% | Valid Acc: {:>6.2f}% | "
                         "T/Epoch: {:<7} | LR: {:>2.6f} | DropPath: {:>1.3f} |".format(
                epoch + 1, self.epochs,
                train_acc * 100, valid_acc * 100,
                show_time((time.time() - t_start) / (epoch + 1)),
                self.scheduler.get_last_lr()[0],
                self.model.drop_path_prob    # ADDED: log drop path prob
            ))
 
            self.scheduler.step()
 
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_train_acc = train_acc
                best_epoch     = epoch + 1
                best_model     = copy.deepcopy(self.model)
 
            if time.time() - t_start > self.train_time_limit - self.extra_time:
                logging.info("Final Training Time Exceeded Given Limit")
                break
 
        logging.info("Total final training runtime: {}".format(show_time(time.time() - t_start)))
        self.model = best_model
        logging.info("Best Model Stats: Epoch {:>3}, Train Acc: {:>6.2f}%, Valid Acc: {:>6.2f}%".format(
            best_epoch, best_train_acc * 100, best_valid_acc * 100
        ))
        log_lines(20)
        return self.model
 
 
    def evaluate(self):
        self.model.eval()
        labels, predictions = [], []
        for data, target in self.valid_dataloader:
            data = data.cuda(non_blocking=True)
 
            # CHANGED: unpack tuple — same fix as in train()
            # logits_aux is None during eval() so safe to ignore
            logits, _ = self.model.forward(data)
 
            labels      += target.cpu().tolist()
            predictions += torch.argmax(logits, 1).detach().cpu().tolist()
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
