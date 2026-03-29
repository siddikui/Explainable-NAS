import torchvision
import torch.nn as nn
import torch
import torch

from torch import optim
# CHANGED: import Network and NetworkCIFAR instead of NetworkMix
from helpers import Network, NetworkCIFAR, show_time, general_num_params, Clock, set_seed, log_lines
import logging, sys, os
from sklearn.metrics import accuracy_score
import numpy as np
import time
import copy
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from collections import Counter
from torchsummary import summary
import torch.nn.functional as F

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
        raise RuntimeError(f"CUDA out of memory: GPU memory usage exceeded {max_mb} MB (used: {used} MB)")


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('CIFAR-ImageNet.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

class NAS:

    def __init__(self, train_loader, valid_loader, metadata, clock):
    
        search_size = 0.1
        
        data = train_loader.dataset.x
        labels = train_loader.dataset.y
        
        train_indices, val_indices = train_test_split(range(len(labels)),
                                                      test_size=1-search_size,
                                                      stratify=labels)
        train_subset = Subset(train_loader.dataset, train_indices)
        val_subset = Subset(train_loader.dataset, val_indices) 
        
        train_loader = torch.utils.data.DataLoader(train_subset, 
                                                   batch_size=64, 
                                                   drop_last=True,
                                                   shuffle=True)
        val_loader_subset = torch.utils.data.DataLoader(val_subset, batch_size=64, shuffle=False)       

        total_valid_size = len(valid_loader.dataset)
        valid_subset_size = total_valid_size // 1
        valid_subset, _ = torch.utils.data.random_split(
            valid_loader.dataset, [valid_subset_size, total_valid_size - valid_subset_size]
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_subset, batch_size=64, drop_last=True, shuffle=False
        )

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

        from helpers import get_per_dataset_time_limits
        if not hasattr(NAS, '_total_time'):
            if hasattr(clock, 'time_limit') and hasattr(clock, 'start_time'):
                NAS._total_time = float(clock.time_limit - clock.start_time)
            else:
                NAS._total_time = 15 * 60

        n_datasets = int(metadata.get('n_datasets', 3))
        dataset_idx = int(metadata.get('dataset_idx', NAS._dataset_counter-1 if hasattr(NAS, '_dataset_counter') else 0))

        total_time = NAS._total_time
        search_time, train_time, extra_time = get_per_dataset_time_limits(
            total_time, dataset_idx, n_datasets, search_ratio=0.5, train_ratio=0.5)
        self.phase1_time_out = False
        self.phase2_time_out = False
        self.search_time_limit = search_time
        self.train_time_limit = train_time
        self.extra_time = extra_time
        self.metadata['train_time_limit'] = self.train_time_limit

        logging.info(f"[NAS] Dataset {dataset_idx+1}/{n_datasets}: Allocated {show_time(search_time+train_time)} "
                     f"(search: {show_time(search_time)}, train: {show_time(train_time)}, extra: {show_time(extra_time)})")

        self.total_time_limit = getattr(self, 'search_time_limit', None)
        self.phase1_time_limit = self.total_time_limit * 0.40 if self.total_time_limit else None
        self.phase1_time_limit -= self.extra_time
        self.phase2_time_limit = self.total_time_limit * 0.60 if self.total_time_limit else None
        self.phase2_time_limit -= self.extra_time

        logging.info(f"Total Search Time: {self.total_time_limit})")
        logging.info(f"Phase 1 Search Time: {self.phase1_time_limit})")
        logging.info(f"Phase 2 Search Time: {self.phase2_time_limit})")

    def get_checkpoint_path(self, layers=None, channels=None, seed=None):
        if layers is not None and channels is not None and seed is not None:
            return f"checkpoints/ckpt_{self.metadata['codename']}_L{layers}_C{channels}_S{seed}.pt"
 
    def save_checkpoint(self, model, epoch, layers, channels, seed):
        print(f"Saving checkpoint for layers={layers}, channels={channels}, seed={seed}")
        path = self.get_checkpoint_path(layers, channels, seed)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'layers':    layers,
            'channels':  channels,
            'seed':      seed,
            'epoch':     epoch,
            'model':     model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            # ADDED: save arch optimizer and alphas so search progress
            # is not lost on resume. Without these, alphas reset to
            # random init and the cell search starts over from scratch.
            'arch_optimizer': self.arch_optimizer.state_dict(),
            'alphas_normal':  model.alphas_normal.data,
            'alphas_reduce':  model.alphas_reduce.data,
        }, path)
        logging.info(f"Checkpoint saved: {path}")
 
    def load_checkpoint_if_exists(self, model, optimizer, scheduler,
                                   layers, channels, seed):
        path = self.get_checkpoint_path(layers, channels, seed)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            logging.info(f"Found checkpoint at {path} with "
                         f"layers={ckpt['layers']}, channels={ckpt['channels']}, seed={ckpt['seed']}")
 
            # UNCHANGED: mismatch checks
            if ckpt['layers'] != layers:
                return 0
            if ckpt['channels'] != channels:
                return 0
 
            # UNCHANGED: restore weights, optimizer, scheduler
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
 
            # ADDED: restore alphas and arch optimizer.
            # alphas_normal and alphas_reduce are not part of model.state_dict()
            # because they are nn.Parameter created via Variable() in DARTS.
            # They must be restored manually.
            if 'alphas_normal' in ckpt:
                model.alphas_normal.data.copy_(ckpt['alphas_normal'])
            if 'alphas_reduce' in ckpt:
                model.alphas_reduce.data.copy_(ckpt['alphas_reduce'])
            if 'arch_optimizer' in ckpt:
                self.arch_optimizer.load_state_dict(ckpt['arch_optimizer'])
 
            model.eval()
            with torch.no_grad():
                resumed_acc = self.evaluate()
            logging.info(f"Resumed model validation accuracy: {resumed_acc*100:.2f}%")
 
            # ADDED: log the genotype at resume so you can see what
            # the search had discovered up to this checkpoint
            logging.info(f"Resumed genotype: {model.genotype()}")
 
            start_epoch = ckpt['epoch']
            return start_epoch
        else:
            logging.info(f"No checkpoint found for layers={layers}, channels={channels}, seed={seed}")
            return 0
 
    def get_checkpoint_if_exists(self, layers, channels, seed):
        # UNCHANGED
        path = self.get_checkpoint_path(layers, channels, seed)
        if os.path.exists(path):
            return torch.load(path, map_location='cpu')
        return None
    

    def train(self, epochs, model, phase_check, layers=None, channels=None, seed=0):

        self.model = model.to(self.device)
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()

        # 1. optimizer first
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.025,
            momentum=0.9,
            weight_decay=3e-4
        )

        # 2. arch_optimizer second
        self.arch_optimizer = optim.Adam(
            self.model.arch_parameters(),
            lr=3e-4,
            betas=(0.5, 0.999),
            weight_decay=1e-3
        )

        # 3. scheduler last — needs self.optimizer to exist
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )

        if layers is not None and channels is not None:
            start_epoch = self.load_checkpoint_if_exists(
                self.model, self.optimizer, self.scheduler,
                layers, channels, seed
            )
        else:
            start_epoch = 0


        if self.device.type == 'cuda':
            logging.info('Running on GPU (CUDA)')
        else:
            logging.info('Running on CPU (worst-case scenario)')

        t_start = time.time()
        best_valid_acc = 0.0
        best_train_acc = 0.0
        best_model = None
        batch_size = self.train_loader.batch_size if hasattr(self.train_loader, 'batch_size') else 16

        # ADDED 3: Persistent iterator over val loader for alpha updates.
        # DARTS needs one val batch per train batch to update alphas.
        # We cycle the val loader so it never runs out mid-epoch.
        val_iter = iter(self.valid_loader)

        try:
            for epoch in range(start_epoch, epochs):

                # ── UNCHANGED time checks ──────────────────────────────────
                if phase_check == 'phase1':
                    if self.phase1_time_limit is not None and (time.time() - self.search_start) > self.phase1_time_limit:
                        logging.info(f"Phase 1 time limit exceeded. Moving to phase 2.")
                        self.phase1_time_out = True
                        break
                elif phase_check == 'phase2':
                    if self.phase2_time_limit is not None and (time.time() - self.search_end) > self.phase2_time_limit:
                        logging.info(f"Phase 2 time limit exceeded. Stopping search.")
                        self.phase2_time_out = True
                        break

                self.model.train()
                labels, predictions = [], []

                for data, target in self.train_loader:
                    data, target = data.to(self.device), target.to(self.device)

                    # ADDED 4: Alpha update step (architecture parameters).
                    # This is the bilevel part of DARTS — alphas are updated
                    # on the validation batch BEFORE weights are updated on
                    # the train batch. Without this block, alphas never change
                    # and genotype() always returns random/equal ops.
                    try:
                        val_data, val_target = next(val_iter)
                    except StopIteration:
                        # Restart val iterator when exhausted
                        val_iter = iter(self.valid_loader)
                        val_data, val_target = next(val_iter)

                    val_data   = val_data.to(self.device)
                    val_target = val_target.to(self.device)

                    self.arch_optimizer.zero_grad()
                    arch_loss = self.model._loss(val_data, val_target)
                    arch_loss.backward()
                    self.arch_optimizer.step()
                    # ── end alpha update ───────────────────────────────────

                    # ── Weight update (same as your original) ──────────────
                    self.optimizer.zero_grad()
                    output = self.model.forward(data)
                    labels      += target.cpu().tolist()
                    predictions += torch.argmax(output, 1).detach().cpu().tolist()
                    loss = self.criterion(output, target)
                    loss.backward()

                    # ADDED 5: Gradient clipping.
                    # DARTS clips weight gradients at norm=5 before optimizer
                    # step. Without this, deep networks with many cells can
                    # have exploding gradients and diverge quickly.
                    nn.utils.clip_grad_norm_(self.model.parameters(), 5)

                    self.optimizer.step()
                    # ── end weight update ──────────────────────────────────

                # ADDED 6: Scheduler step moved INSIDE epoch loop.
                # Your original had it commented out. DARTS steps the cosine
                # scheduler every epoch so lr decays from 0.025 to near zero.
                self.scheduler.step()

                train_acc = accuracy_score(labels, predictions)
                valid_acc = self.evaluate()

                # ADDED 7: Log alpha weights so you can see search progress.
                # Prints the softmax of alphas_normal each epoch so you can
                # watch which ops are winning. Remove if too verbose.
                logging.info("alphas_normal = %s",
                    F.softmax(self.model.alphas_normal, dim=-1).data.cpu().numpy().round(2))

                logging.info("\tEpoch {:>3}/{:<3} | Train Acc: {:>6.2f}% | Valid Acc: {:>6.2f}% | "
                             "T/Epoch: {:<7} | LR: {:>2.6f} | Arch Loss: {:>2.4f} |".format(
                    epoch + 1, self.epochs,
                    train_acc * 100, valid_acc * 100,
                    show_time((time.time() - t_start) / (epoch + 1)),
                    self.scheduler.get_last_lr()[0],
                    arch_loss.item()          # ADDED 8: log arch loss alongside train loss
                ))

                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    best_train_acc = train_acc
                    self.best_epoch = epoch + 1
                    best_model = copy.deepcopy(self.model)
                    self.save_checkpoint(self.model, epoch + 1, layers, channels, seed)

        except RuntimeError as e:
            if 'CUDA out of memory' in str(e) or 'memory' in str(e):
                logging.warning(f"CUDA OOM at batch size {batch_size}.")
                torch.cuda.empty_cache()
                return 0.0, 0.0
            else:
                raise

        self.model = best_model
        logging.info("Candidate Evaluation Time: {}".format(show_time(time.time() - t_start)))
        return best_train_acc * 100, best_valid_acc * 100


    def search_depth_and_width(self):
        self.search_start = time.time()

        logging.info('RUNNING SEARCH on %s', self.metadata['codename'])

        max_params = 4_200_000

        if self.metadata['input_shape'][2] > 32 or self.metadata['input_shape'][3] > 32:
            width_resolution = 48
        else:
            width_resolution = 64

        target_acc = 100
        min_width = 16
        max_width = 2048
        depth_resolution = 2
        min_depth = 2
        max_depth = 10
        max_epochs = 5
        Rand_train = 2
        max_models = 3
        candidate_count = 0

        r1_thresh = 0.25
        r2_thresh = 0.10

        channels = f_channels = min_width
        layers = min_depth

        add_epochs = 1
        s_epoch = epochs = 1
        f_epochs = 0

        macro_count = 0

        bst_dep = []
        bst_wdt = []
        bst_epc = []
        bst_tac = []
        bst_vac = []
        bst_prm = []

        # CHANGED: removed curr_arch_ops, curr_arch_kernel
        # replaced with curr_genotype, best_genotype
        curr_genotype = None
        best_genotype = None

        curr_arch_train_acc = next_arch_train_acc = 0.0
        curr_arch_test_acc  = next_arch_test_acc  = 0.0

        # ── Baseline ─────────────────────────────────────────────────────
        # CHANGED: Network(channels, 10, layers, criterion) 
        #          instead of NetworkMix(channels, metadata, layers, ops, kernels)
        criterion = nn.CrossEntropyLoss().to(self.device)
        model = Network(channels, 10, layers, criterion)
        logging.info("Model Depth %s Model Width %s Train Epochs %s", layers, channels, epochs)
        logging.info("Model Parameters = %f", general_num_params(model))
        logging.info('Evaluating Baseline Model...')
        summary(
            model.to(self.device),
            input_size=( 3, 32, 32),
            device=str(self.device)
        )
        curr_arch_train_acc, curr_arch_test_acc = self.train(epochs, model, 'baseline', layers, channels, 0)
        # self.save_checkpoint(model, epochs)
        logging.info("Baseline Train Acc %f Baseline Val Acc %f", curr_arch_train_acc, curr_arch_test_acc)

        # CHANGED: extract genotype from baseline model
        curr_genotype = model.genotype()
        best_genotype = curr_genotype
        logging.info('Baseline genotype = %s', curr_genotype)

        # ── Phase 1 ───────────────────────────────────────────────────────
        layers_up = channels_up = epochs_up = True

        while curr_arch_test_acc < target_acc:
            torch.cuda.empty_cache()
            if self.phase1_time_out:
                self.phase1_time_out = False
                break
            torch.cuda.empty_cache()

            if curr_arch_train_acc >= 99:
                break

            if (layers >= max_depth and channels >= max_width) or epochs >= max_epochs:
                break

            if layers_up and layers < max_depth:
                layers += depth_resolution
                channels += int(width_resolution / 2)
                layers_up = False
                channels_up = True
                macro_count += 1

            elif channels_up and channels < max_width:
                channels += int(width_resolution / 2)
                channels_up = False
                epochs_up = True
                macro_count += 1

            elif epochs_up and epochs < max_epochs:
                epochs = epochs + add_epochs
                epochs_up = False
                layers_up = True
                channels_up = True

            logging.info('#############################################################################')
            logging.info('Moving to Next Candidate Architecture...')
            candidate_count += 1
            logging.info('Candidate count: %f', candidate_count)

            if candidate_count > max_models:
                logging.info("Maximum Models Evaluated")
                break

            logging.info("Model Depth %s Model Width %s Train Epochs %s", layers, channels, epochs)

            archbestt, archbestv = 0.0, 0.0

            # CHANGED: Network instead of NetworkMix for param check
            criterion = nn.CrossEntropyLoss().to(self.device)
            model = Network(channels, 10, layers, criterion)
            num_params = general_num_params(model)
            logging.info("Model Parameters = %f", num_params)

            if num_params > max_params:
                logging.info("Model Parameters Exceed Upper Bound")
                break

            for i in range(Rand_train):
                torch.cuda.empty_cache()
                set_seed(i)
                logging.info("INITIALIZING RUNNING RUN %f", i)

                # CHANGED: Network instead of NetworkMix
                model = Network(channels, 10, layers, nn.CrossEntropyLoss().to(self.device))
                next_arch_train_acc, next_arch_test_acc = self.train(epochs, model, 'phase1', layers, channels, i)

                if next_arch_test_acc == 0.0 and next_arch_train_acc == 0.0:
                    break
                if next_arch_test_acc > archbestv:
                    archbestt = next_arch_train_acc
                    archbestv = next_arch_test_acc
                    model = self.model
                    # CHANGED: save genotype of best run
                    best_genotype = model.genotype()

                logging.info("Candidate Train Acc %f Candidate Val Acc %f",
                             next_arch_train_acc, next_arch_test_acc)

            if next_arch_test_acc == 0.0 and next_arch_train_acc == 0.0:
                break

            next_arch_train_acc = archbestt
            next_arch_test_acc  = archbestv
            logging.info("Candidate Best Train %f Candidate Best Val %f",
                         next_arch_train_acc, next_arch_test_acc)

            if next_arch_test_acc > curr_arch_test_acc + r1_thresh:
                if channels_up is True and layers_up is False:
                    layers_up = True
                elif channels_up is False and layers_up is False:
                    channels_up = True

                logging.info("Train Acc Diff %f Val Acc Diff %f",
                             next_arch_train_acc - curr_arch_train_acc,
                             next_arch_test_acc  - curr_arch_test_acc)
                curr_arch_train_acc = next_arch_train_acc
                curr_arch_test_acc  = next_arch_test_acc
                f_channels = channels
                f_epochs   = epochs
                s_epoch    = self.best_epoch

                # CHANGED: update accepted genotype
                curr_genotype = best_genotype

                # self.save_checkpoint(model, self.best_epoch)
                logging.info("Highest Train Acc %f Highest Val Acc %f",
                             curr_arch_train_acc, curr_arch_test_acc)

                bst_dep.append(layers)
                bst_wdt.append(channels)
                bst_epc.append(epochs)
                bst_tac.append(round(next_arch_train_acc, 2))
                bst_vac.append(round(next_arch_test_acc,  2))
                bst_prm.append(general_num_params(model))

                logging.info('Best Arch Layers: %s',    bst_dep)
                logging.info('Best Arch Channels: %s',  bst_wdt)
                logging.info('Best Arch Epochs: %s',    bst_epc)
                logging.info('Best Arch Train Acc: %s', bst_tac)
                logging.info('Best Arch Val Acc: %s',   bst_vac)
                logging.info('Best Arch Params: %s',    bst_prm)

            else:
                logging.info("Highest Train Acc %f Highest Val Acc %f",
                             curr_arch_train_acc, curr_arch_test_acc)
                logging.info("Train Acc Diff %f Val Acc Diff %f",
                             next_arch_train_acc - curr_arch_train_acc,
                             next_arch_test_acc  - curr_arch_test_acc)
                continue

        # CHANGED: f_layers from layers directly (no curr_arch_ops array)
        f_layers = layers

        logging.info('Discovered Depth %s',   f_layers)
        logging.info('Discovered Width %s',   f_channels)
        logging.info('Discovered Epochs %s best saved epoch %s', f_epochs, s_epoch)
        logging.info('Discovered Genotype %s', curr_genotype)

        logging.info('#############################################################################')
        logging.info('Best Arch Layers: %s',    bst_dep)
        logging.info('Best Arch Channels: %s',  bst_wdt)
        logging.info('Best Arch Epochs: %s',    bst_epc)
        logging.info('Best Arch Train Acc: %s', bst_tac)
        logging.info('Best Arch Val Acc: %s',   bst_vac)
        logging.info('Best Arch Params: %s',    bst_prm)
        logging.info('#############################################################################')

        # ── Phase 2 ───────────────────────────────────────────────────────
        Rand_train = 5
        candidate_count = 0

        self.search_end = time.time()
        phase1_time = self.search_end - self.search_start
        logging.info(f"Phase 1 Search Time: {phase1_time})")

        self.phase2_time_limit = self.total_time_limit - phase1_time
        self.phase2_time_limit -= self.extra_time
        logging.info(f"Phase 2 Allocated Time: {self.phase2_time_limit})")

        while len(bst_dep) > 1:
            if self.phase2_time_out:
                self.phase2_time_out = False
                break

            bst_dep_2 = []
            bst_wdt_2 = []
            bst_epc_2 = []
            bst_tac_2 = []
            bst_vac_2 = []
            bst_prm_2 = []

            if candidate_count > max_models:
                break

            for i in range(len(bst_dep)):
                layers   = bst_dep[len(bst_dep) - 2 - i]
                channels = bst_wdt[len(bst_wdt) - 2 - i]

                if i == 0:
                    epochs = max(f_epochs, bst_epc[len(bst_epc) - 1]) + 1
                else:
                    epochs = epochs + 1

                archbestt, archbestv = 0.0, 0.0

                # CHANGED: Network instead of NetworkMix for param log
                criterion = nn.CrossEntropyLoss().to(self.device)
                model = Network(channels, 10, layers, criterion)

                logging.info('Moving to Next Candidate Architecture...')
                logging.info("Model Depth %s Model Width %s Train Epochs %s", layers, channels, epochs)
                logging.info("Model Parameters = %f", general_num_params(model))

                for j in range(Rand_train):
                    torch.cuda.empty_cache()
                    if self.phase2_time_limit is not None and \
                       (time.time() - self.search_end) > self.phase2_time_limit:
                        logging.info(f"Phase 2 time limit exceeded. Stopping.")
                        break

                    set_seed(j)
                    logging.info("INITIALIZING RUNNING RUN %f", j)

                    # CHANGED: Network instead of NetworkMix
                    model = Network(channels, 10, layers, nn.CrossEntropyLoss().to(self.device))
                    next_arch_train_acc, next_arch_test_acc = self.train(epochs, model, 'phase2', layers, channels, j)

                    if next_arch_test_acc > archbestv:
                        archbestt = next_arch_train_acc
                        archbestv = next_arch_test_acc
                        model = self.model
                        # CHANGED: save genotype of best run
                        best_genotype = model.genotype()

                    logging.info("Candidate Train Acc %f Candidate Val Acc %f",
                                 next_arch_train_acc, next_arch_test_acc)

                next_arch_train_acc = archbestt
                next_arch_test_acc  = archbestv
                logging.info("Candidate Best Train %f Candidate Best Val %f",
                             next_arch_train_acc, next_arch_test_acc)

                candidate_count += 1

                if next_arch_test_acc > curr_arch_test_acc + r2_thresh:
                    # self.save_checkpoint(model, self.best_epoch)

                    f_layers   = layers
                    f_channels = channels
                    f_epochs   = epochs

                    # CHANGED: update accepted genotype
                    curr_genotype = best_genotype

                    logging.info("Candidate Train Acc %f Candidate Val Acc %f",
                                 next_arch_train_acc, next_arch_test_acc)
                    logging.info("Highest Train Acc %f Highest Val Acc %f",
                                 curr_arch_train_acc, curr_arch_test_acc)

                    curr_arch_train_acc = next_arch_train_acc
                    curr_arch_test_acc  = next_arch_test_acc

                    bst_dep_2.append(layers)
                    bst_wdt_2.append(channels)
                    bst_epc_2.append(epochs)
                    bst_tac_2.append(round(next_arch_train_acc, 2))
                    bst_vac_2.append(round(next_arch_test_acc,  2))
                    bst_prm_2.append(general_num_params(model))

                    logging.info('Best Arch Layers: %s',    bst_dep_2)
                    logging.info('Best Arch Channels: %s',  bst_wdt_2)
                    logging.info('Best Arch Epochs: %s',    bst_epc_2)
                    logging.info('Best Arch Train Acc: %s', bst_tac_2)
                    logging.info('Best Arch Val Acc: %s',   bst_vac_2)
                    logging.info('Best Arch Params: %s',    bst_prm_2)

                else:
                    logging.info("Candidate Train Acc %f Candidate Val Acc %f",
                                 next_arch_train_acc, next_arch_test_acc)
                    logging.info("Highest Train Acc %f Highest Val Acc %f",
                                 curr_arch_train_acc, curr_arch_test_acc)

                logging.info('#############################################################################')

            bst_dep = bst_dep_2
            bst_wdt = bst_wdt_2
            bst_epc = bst_epc_2
            bst_tac = bst_tac_2
            bst_vac = bst_vac_2
            bst_prm = bst_prm_2

            bst_dep, bst_wdt, bst_epc, bst_tac, bst_vac, bst_prm = self.sort_networks(
                bst_dep, bst_wdt, bst_epc, bst_tac, bst_vac, bst_prm)

            logging.info('Best Arch Layers: %s',    bst_dep)
            logging.info('Best Arch Channels: %s',  bst_wdt)
            logging.info('Best Arch Epochs: %s',    bst_epc)
            logging.info('Best Arch Train Acc: %s', bst_tac)
            logging.info('Best Arch Val Acc: %s',   bst_vac)
            logging.info('Best Arch Params: %s',    bst_prm)

        logging.info('Discovered Final Depth %s',    f_layers)
        logging.info('Discovered Final Width %s',    f_channels)
        logging.info('Discovered Final Epochs %s',   f_epochs)
        logging.info('Discovered Final Genotype %s', curr_genotype)

        log_lines(10)

        phase2_time_taken = time.time() - self.search_end
        logging.info(f"Phase 2 Taken Time: {phase2_time_taken})")

        # CHANGED: return genotype instead of ops/kernel arrays
        return curr_genotype, f_channels, f_layers


    def evaluate(self):
        # UNCHANGED
        self.model.eval()
        labels, predictions = [], []
        for data, target in self.valid_loader:
            data = data.cuda(non_blocking=True)
            output = self.model.forward(data)
            labels += target.cpu().tolist()
            predictions += torch.argmax(output, 1).detach().cpu().tolist()
        return accuracy_score(labels, predictions)

    # def save_checkpoint(self, model, epoch):
    #     # UNCHANGED
    #     torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #     }, self.metadata["codename"]+".pth")
    #     print(f"Checkpoint saved to {self.metadata['codename']}.pth")

    def search(self):
        # CHANGED: 3 return values, NetworkCIFAR with genotype
        genotype, f_channels, f_layers = self.search_depth_and_width()

        logging.info('Final Genotype:  %s', genotype)
        logging.info('Final Channels:  %s', f_channels)
        logging.info('Final Layers:    %s', f_layers)

        # CHANGED: NetworkCIFAR with genotype instead of NetworkMix with ops/kernels
        model = NetworkCIFAR(f_channels, 10, f_layers, auxiliary=False, genotype=genotype)
        return model

    def sort_networks(self, d, w, e, t, v, p):
        # UNCHANGED
        if len(d) == 0:
            return d, w, e, t, v, p
        combined = list(zip(d, w, e, t, v, p))
        combined_sorted = sorted(combined, key=lambda x: x[5])
        d, w, e, t, v, p = zip(*combined_sorted)
        return list(d), list(w), list(e), list(t), list(v), list(p)