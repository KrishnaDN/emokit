from __future__ import print_function

import argparse
import copy
from emokit.dataset.dataset import PreProcessor
import logging
import os
import warnings
import torch
import torch.distributed as dist
import torch.optim as optim
import numpy as np
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from emokit.dataset import KaldiDataset, CollateFunc, PreProcessor
from emokit.models import AudioModels, TextModels, MultiModalModels
from emokit.utils import load_checkpoint, save_checkpoint, average_models
from emokit.bin import BuildOptimizer, BuildScheduler
from emokit.bin import Executor

warnings.filterwarnings("ignore")

class Trainer:
    def __init__(self, params, args):
        self.params = params
        self.args = args
    
    @property
    def initialize_model(self,):
        if self.params['model']['audio_only']:
            model = AudioModels[self.params['model']['audio_model_settings']['model_type']](self.params['model']['audio_model_settings']['model_params'])
        elif self.params['model']['audio_only']:
            raise NotImplementedError
        else:
            raise NotImplementedError
        return model
    

    def train(self,):    
        data_file = self.params['data']['scp_file']
        text_file = self.params['data']['text_file']
        labels = self.params['data']['labels']
        cmvn_file = self.params['data']['cmvn_file']
        dataset_conf = self.params['dataset_conf']['kaldi_offline_conf']
        processor = PreProcessor(data_file, text_file, cmvn_file, labels)
        train, test = processor._split_data
        train_dataset = KaldiDataset(train, cmvn_file,)
        cv_dataset = KaldiDataset(test, cmvn_file,)
            
        distributed = self.args.world_size > 1
        if distributed:
            logging.info('training on multiple gpus, this gpu {}'.format(self.args.gpu))
            dist.init_process_group(self.args.dist_backend,
                                    init_method=self.args.init_method,
                                    world_size=self.args.world_size,
                                    rank=self.args.rank)
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, shuffle=True)
            cv_sampler = torch.utils.data.distributed.DistributedSampler(
                cv_dataset, shuffle=False)
        else:
            train_sampler = None
            cv_sampler = None

        collate_fun = CollateFunc(**dataset_conf, mode='train')
        train_data_loader = DataLoader(train_dataset,
                                    collate_fn=collate_fun,
                                    sampler=train_sampler,
                                    shuffle=False,
                                    num_workers=self.args.num_workers,
                                    batch_size=self.params['train']['batch_size'])
        
        cv_collate_conf = copy.deepcopy(dataset_conf)
        cv_collate_fun = CollateFunc(**cv_collate_conf, mode='test')
        
        cv_data_loader = DataLoader(cv_dataset,
                                    collate_fn=cv_collate_fun,
                                    sampler=cv_sampler,
                                    shuffle=False,
                                    num_workers=self.args.num_workers,
                                    batch_size=self.params['train']['batch_size'])

        test_data_loader = DataLoader(cv_dataset,
                                    collate_fn=cv_collate_fun,
                                    sampler=cv_sampler,
                                    shuffle=False,
                                    num_workers=self.args.num_workers,
                                    batch_size=1)
        
        model = self.initialize_model
        print(model)
        executor = Executor()
        if self.args.checkpoint is not None:
            infos = load_checkpoint(model, self.args.checkpoint)
        else:
            infos = {}
        start_epoch = infos.get('epoch', -1) + 1
        cv_loss = infos.get('cv_loss', 0.0)
        step = infos.get('step', -1)

        num_epochs = self.params['train']['epochs']
        model_dir = os.path.join(self.params['train']['exp_dir'],self.params['train']['model_dir'])
        writer = None
        if self.args.rank == 0:
            os.makedirs(model_dir, exist_ok=True)
            exp_id = os.path.basename(model_dir)
            writer = SummaryWriter(os.path.join(model_dir, exp_id))

        if distributed:
            assert (torch.cuda.is_available())
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=True)
            device = torch.device("cuda")
        else:
            use_cuda = self.args.gpu >= 0 and torch.cuda.is_available()
            device = torch.device('cuda' if use_cuda else 'cpu')
            model = model.to(device)

        optimizer = BuildOptimizer[self.params['train']['optimizer_type']](
            filter(lambda p: p.requires_grad, model.parameters()), **self.params['train']['optimizer']
        )
        scheduler = BuildScheduler[self.params['train']['scheduler_type']](optimizer, **self.params['train']['scheduler'])
        final_epoch = None
        self.params['rank'] = self.args.rank
        self.params['is_distributed'] = distributed
        if start_epoch == 0 and self.args.rank == 0:
            save_model_path = os.path.join(model_dir, 'init.pt')
            save_checkpoint(model, save_model_path)
        
        executor.step = step
        scheduler.step()
        for epoch in range(start_epoch, num_epochs):
            if distributed:
                train_sampler.set_epoch(epoch)
            lr = optimizer.param_groups[0]['lr']
            logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
            executor.train(model, optimizer, scheduler, train_data_loader, device,
                        writer, self.params)
            total_loss, num_seen_utts, test_acc = executor.cv(model, cv_data_loader, device,
                                                    self.params)
            
            
            if self.args.world_size > 1:
                # all_reduce expected a sequence parameter, so we use [num_seen_utts].
                num_seen_utts = torch.Tensor([num_seen_utts]).to(device)
                # the default operator in all_reduce function is sum.
                dist.all_reduce(num_seen_utts)
                total_loss = torch.Tensor([total_loss]).to(device)
                dist.all_reduce(total_loss)
                cv_loss = total_loss[0] / num_seen_utts[0]
                cv_loss = cv_loss.item()
            else:
                cv_loss = total_loss / num_seen_utts

            logging.info('Epoch {} CV info cv_loss {} accuracy {}'.format(epoch, cv_loss, test_acc))
            if self.args.rank == 0:
                save_model_path = os.path.join(model_dir, '{}.pt'.format(epoch))
                save_checkpoint(
                    model, save_model_path, {
                        'epoch': epoch,
                        'lr': lr,
                        'cv_loss': cv_loss,
                        'step': executor.step,
                        'acc': test_acc
                    })
                writer.add_scalars('epoch', {'cv_loss': cv_loss, 'lr': lr}, epoch)
            final_epoch = epoch

        if final_epoch is not None and self.args.rank == 0:
            final_model_path = os.path.join(model_dir, 'final.pt')
            os.symlink('{}.pt'.format(final_epoch), final_model_path)
        
    
    

    
