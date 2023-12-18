# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime
import logging
import wandb
import signal

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, download_dataset

from custom_comm import tcp_create_group, tcp_allreduce, tcp_destroy_group

# import wecloud_callback

# logger = logging.Logger(__name__)

def do_log(entries):
    wandb.log({x:y for x,y in entries})
    logging.info(", ".join([f"{x} = {y}" for x,y in entries]))
    # csv_writer.write("{},{},{},{},{},{},{}".format(
    #     epoch,                                  # epoch
    #     n_iter,                                 # iteration
    #     batch_index * args.b + len(images),     # trained_samples
    #     len(cifar100_training_loader.dataset),  # total_samples
    #     loss.item(),                            # loss
    #     optimizer.param_groups[0]['lr'],        # lr
    #     time.time() - epoch_start_time,         # current epoch wall-clock time
    # ))

def train_once(inputs, targets):
    if args.gpu:
        targets = targets.to(device)
        inputs = inputs.to(device)
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss, outputs


def train(epoch):

    training_sampler.set_epoch(epoch) #yyt
    net.train()

    epoch_start_time = time.time()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        # wecloud_callback.step_begin()

        loss, outputs = train_once(images, labels)

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        # last_layer = list(net.children())[-1]
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        #yyt
        do_log([("iteration", n_iter), 
                ("loss", loss.item()),
                ("lr", optimizer.param_groups[0]['lr']),
                ])

        if args.profiling:
            logging.info(f"PROFILING: dataset total number {len(cifar100_training_loader.dataset)}, training one batch costs {time.time() - batch_start_time} seconds")
            return

        #update training loss for each iteration
        # writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm: # epoch starts from 1
            scheduler.step() # call inner warmup scheduler
        # wecloud_callback.step_end()

    if epoch > args.warm:
        scheduler.step() # call inner training scheduler
    
    # for name, param in net.named_parameters():
    #     layer, attr = os.path.splitext(name)
    #     attr = attr[1:]
    #     writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    epoch_finish_time = time.time()

    logging.info('epoch {} training time consumed: {:.2f}s'.format(epoch, epoch_finish_time - epoch_start_time))

@torch.no_grad()
def validate(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.to(device)
            labels = labels.to(device)

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        logging.info('GPU INFO.....')
        logging.info(torch.cuda.memory_summary())
    logging.info('Evaluating Network.....')
    logging.info('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader), # loss has been averaged
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    wandb.log({"epoch"     : epoch, 
            "test_loss"    : test_loss / len(cifar100_test_loader),  
            "test_accuracy": correct.float() / len(cifar100_test_loader.dataset)})
    #add informations to tensorboard
    # if tb:
    #     writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
    #     writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)   

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True, help='net type')
    parser.add_argument('--gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('--epoch', type=int, default=settings.EPOCH, help='num of epochs to train')
    parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--target_lr', type=float, default=0.001, help='target learning rate (only for expLR)')
    parser.add_argument('--resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--profiling', action="store_true", default=False, help="profile one batch")
    parser.add_argument('--comm_hook', action="store_true", default=False, help="use custom communication hook")
    args = parser.parse_args()

    wandb.init(
        # set the wandb project where this run will be logged
        project="ML",
        
        # track hyperparameters and run metadata
        config={
            "model": args.net,
            "dataset": "CIFAR-100",
            "epochs": args.epoch,
            "batch_size": args.b,
            "learning_rate": args.lr,
            "target_learning_rate": args.target_lr
        }
    )

# set distributed backend
    dist.init_process_group(backend="nccl") # If neither is specified, init_method is assumed to be “env://”.

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    # torch.cuda.set_device(local_rank)
    device = torch.device("cuda:{}".format(local_rank))

# get model
    net = get_network(args) # return a pure NN
    net = net.to(device)
    # net = DDP(net, device_ids=[local_rank], output_device=local_rank)
    net = DDP(net, [local_rank])
    if args.comm_hook:
        tcp_group = tcp_create_group()
        def signal_handler(signum, frame):
            tcp_destroy_group(tcp_group)
            exit(1)
        signal.signal(signal.SIGINT, signal_handler)
        net.register_comm_hook(tcp_group, tcp_allreduce)


# set loss function
    loss_function = nn.CrossEntropyLoss()

# set optmizer
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# data processing
    if local_rank == 0:
        download_dataset()
    
    # wait until the dataset is downloaded
    dist.barrier()
    
    cifar100_training_loader, training_sampler = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b
    )

# set scheduler
    # this is a per epoch scheduler
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    # train_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, (args.target_lr/args.lr)**(1/args.epoch)) 
    iter_per_epoch = len(cifar100_training_loader)
    warmup_iters = iter_per_epoch * args.warm
    # this is a per batch scheduler
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1/(warmup_iters+1), total_iters=warmup_iters) # == [base_lr * i/(warmup_iters+1) for i in range(1, warmup_iters+1)] + [base_lr]*n  
    # call per batch during warming up, call per epoch after that
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, train_scheduler], milestones=[warmup_iters])

    print("lr", optimizer.param_groups[0]['lr'])
    # log_header = ["epoch", "trained_samples", "total_samples", "loss", "lr", "current epoch wall-clock time"]
    
    # os.makedirs(os.path.join("logs", args.net), exist_ok=True)
    # csv_path = os.path.join("logs", args.net, f"{settings.TIME_NOW}.csv")
    # csv_writer = open(csv_path, "w")
    # csv_writer.write("epoch,iteration,trained_samples,total_samples,loss,lr,current epoch wall-clock time\n")


    # checkpoint_path = '/app/output/checkpoint'
    # log_dir = '/app/output/log'

    # if args.resume:
    #     recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
    #     if not recent_folder:
    #         raise Exception('no recent folder were found')

    #     checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    # else:
    #     checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    # if not os.path.exists(log_dir):
    #     os.mkdir(log_dir)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    # writer = SummaryWriter(log_dir=os.path.join(log_dir, settings.TIME_NOW))

    # writer.add_graph(net.module) # yyt: I don't know how to modify this

    #create checkpoint folder to save model
    # if not os.path.exists(checkpoint_path):
    #     os.makedirs(checkpoint_path)
    # checkpoint_epoch_path = os.path.join(checkpoint_path, '{epoch}')
    # checkpoint_file_path = os.path.join(checkpoint_epoch_path, 'chkpt.pth')

    # resume_epoch = None

    # best_acc = 0.0
    # if args.resume:
    #     resume_epoch = last_epoch(checkpoint_path)
    #     net.load_state_dict(torch.load(checkpoint_file_path.format(epoch=resume_epoch)))

        # best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        # if best_weights:
        #     weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
        #     logging.info('found best acc weights file:{}'.format(weights_path))
        #     logging.info('load best training file to test acc...')
        #     net.load_state_dict(torch.load(weights_path))
        #     best_acc = eval_training(tb=False)
        #     logging.info('best acc is {:0.2f}'.format(best_acc))

        # recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        # if not recent_weights_file:
        #     raise Exception('no recent weights file were found')
        # weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        # logging.info('loading weights file {} to resume training.....'.format(weights_path))
        # net.load_state_dict(torch.load(weights_path))

        # resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))

    # wecloud_callback.init(
    #     total_steps=args.epoch * iter_per_epoch
    # )
    for epoch in range(1, args.epoch + 1):

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)

        if args.profiling:
            break

        if rank == 0:
            validate(epoch)
        

        #start to save best performance model after learning rate decay to 0.01
        # if epoch > settings.MILESTONES[1] and best_acc < acc:
        #     weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
        #     logging.info('saving weights file to {}'.format(weights_path))
        #     torch.save(net.state_dict(), weights_path)
        #     best_acc = acc
        #     continue

        # if not epoch % settings.SAVE_EPOCH:
        #     weights_dir = checkpoint_epoch_path.format(epoch=epoch)
        #     weights_path = checkpoint_file_path.format(epoch=epoch)
        #     logging.info('saving weights file to {}'.format(weights_path))
        #     if not os.path.exists(weights_dir):
        #         os.makedirs(weights_dir)
        #     torch.save(net.state_dict(), weights_path)
    if args.comm_hook:
        tcp_destroy_group(tcp_group)

    # writer.close()
    # csv_writer.close()

    # df = pd.DataFrame(all_log, columns=log_header)
    # os.makedirs(os.path.join("logs", args.net), exist_ok=True)
    # df.to_csv(os.path.join("logs", args.net, f"{settings.TIME_NOW}.csv"))

