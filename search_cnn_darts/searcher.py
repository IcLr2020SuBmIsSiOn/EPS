import os
import sys
import torch
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
import numpy as np
import PIL
from PIL import Image
import time
import logging
import argparse

from model_search import Network

from utils import get_structure,accuracy, AvgrageMeter, CrossEntropyLabelSmooth, save_checkpoint, get_lastest_model, get_parameters
from evo_controller import *

import random
import pickle
import json
import copy
import torch.backends.cudnn as cudnn


class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]
  
    
def get_args():
    parser = argparse.ArgumentParser("EVOVIN")
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--total-iters', type=int, default=80000, help='total iters')
    parser.add_argument('--warmup-iters', type=int, default=0, help='total iters')
    
    parser.add_argument('--learning-rate', type=float, default=0.1, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    
    parser.add_argument('--evo-momentum', type=float, default=0., help='momentum')
    
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--label-smooth', type=float, default=0.1, help='label smoothing')

    parser.add_argument('--auto-continue', type=bool, default=False, help='report frequency')
    parser.add_argument('--display-interval', type=int, default=10, help='report frequency')
    parser.add_argument('--val-interval', type=int, default=200, help='validation intervals')
    parser.add_argument('--val-times', type=int, default=200, help='valitation times')
    
    parser.add_argument('--p_opwise', type=float, default=0.2, help='opwise probability')
    parser.add_argument('--p_edgewise', type=float, default=0.1, help='edgewise probability')
    
    parser.add_argument('--stacks', type=int, default=8, help='stacks')
    parser.add_argument('--init-channels', type=int, default=16, help='channels')
    
    
    parser.add_argument('--save-interval', type=int, default=400, help='report frequency')

    
    parser.add_argument('--max-population', type=int, default=512, help='max population')
    parser.add_argument('--select-number', type=int, default=64, help='sel number')
    parser.add_argument('--mutation-len', type=int, default=8, help='mul number')
    parser.add_argument('--mutation-number', type=int, default=1, help='mul number')
    
    
    parser.add_argument('--rand-seed', type=int, default=444, help='mul number')
    
    parser.add_argument('--gpu-no', type=str, default='1', help='gpu')
    
    args = parser.parse_args()
    return args

def main():
    
    
    #LOAD CONFIGS################################################################
    args = get_args()
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no
    
    
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    fh = logging.FileHandler(os.path.join('log/train-{}{:02}{}'.format(local_time.tm_year % 2000, local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    
    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
    
    cudnn.benchmark = True
    torch.manual_seed(args.rand_seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.rand_seed)
    
#     cudnn.enabled=True
#     torch.cuda.manual_seed(str(args.rand_seed))
    random.seed(args.rand_seed) 
    #LOAD DATA###################################################################
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std  = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)]
    transform_train = transforms.Compose(lists)
    transform_test  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    def convert_param(original_lists):
      ctype, value = original_lists[0], original_lists[1]
      is_list = isinstance(value, list)
      if not is_list: value = [value]
      outs = []
      for x in value:
        if ctype == 'int':
          x = int(x)
        elif ctype == 'str':
          x = str(x)
        elif ctype == 'bool':
          x = bool(int(x))
        elif ctype == 'float':
          x = float(x)
        elif ctype == 'none':
          if x.lower() != 'none':
            raise ValueError('For the none type, the value must be none instead of {:}'.format(x))
          x = None
        else:
          raise TypeError('Does not know this type : {:}'.format(ctype))
        outs.append(x)
      if not is_list: outs = outs[0]
      return outs
    from collections import namedtuple
    with open('../data/cifar-split.txt', 'r') as f:
        data = json.load(f)
        content = { k: convert_param(v) for k,v in data.items()}
        Arguments = namedtuple('Configure', ' '.join(content.keys()))
        content   = Arguments(**content)
    cifar_split = content
    train_split, valid_split = cifar_split.train, cifar_split.valid
    print(len(train_split),len(valid_split))
    
    train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split),
        num_workers=4, pin_memory=use_gpu)
    
    train_dataprovider = DataIterator(train_loader)
    
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_test),
        batch_size=250, shuffle=False, sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split),
        num_workers=4, pin_memory=use_gpu
    )
    
    val_dataprovider = DataIterator(val_loader)
    print('load data successfully')
    
    model = Network(args.init_channels,  10, args.stacks).cuda()
    
    
    optimizer = torch.optim.SGD(get_parameters(model),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    criterion_smooth = CrossEntropyLabelSmooth(10, 0.1)
    
    if use_gpu:

        loss_function = criterion_smooth.cuda()
        device = torch.device("cuda")
        
    else:
        
        loss_function = criterion_smooth
        device = torch.device("cpu")
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.total_iters)
    model = model.to(device)
    
    all_iters = 0
    if args.auto_continue:
        lastest_model, iters = get_lastest_model()
        if lastest_model is not None:
            all_iters = iters
            checkpoint = torch.load(lastest_model, map_location=None if use_gpu else 'cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print('load from checkpoint')
            for i in range(iters):
                scheduler.step()

    args.optimizer = optimizer
    args.loss_function = loss_function
    args.scheduler = scheduler
    args.train_dataprovider = train_dataprovider
    args.val_dataprovider = val_dataprovider
    
    args.evo_controller = evolutionary(args.max_population,args.select_number, args.mutation_len,args.mutation_number,args.p_edgewise,args.p_opwise)
    
    
    
    
    
    path = './record_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.stacks,args.init_channels,args.total_iters,args.warmup_iters,args.max_population,args.select_number,args.mutation_len,args.mutation_number,args.val_interval,args.val_times,args.p_edgewise,args.p_opwise,args.evo_momentum,args.rand_seed)
    
    logging.info(path)
    
#     args.evo_controller.trained_group = args.evo_controller.group
    
    while all_iters < args.total_iters:
        
        
        if all_iters > 1 and all_iters%args.val_interval == 0:
            results = []
            for structure_father in args.evo_controller.group:
                results.append([structure_father.structure,structure_father.loss,structure_father.count])
            if not os.path.exists(path):
                os.mkdir(path)
                
            with open(path + '/%06d-ep.txt'%all_iters,'w') as tt:
                json.dump(results,tt)
            
            if all_iters >= args.warmup_iters:#warmup
                args.evo_controller.select()

            else:
                print("warmup")
            
            
            
        all_iters = train(model, device, args, val_interval=args.val_interval, bn_process=False, all_iters=all_iters)
        validate(model, device, args, all_iters=all_iters)
    results = []
    for structure_father in args.evo_controller.group:
        results.append([structure_father.structure,structure_father.loss,structure_father.count])
    with open(path + '/%06d-ep.txt'%all_iters,'w') as tt:
        json.dump(results,tt)
    


def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters

def train(model, device, args, *, val_interval, bn_process=False, all_iters=None):
    
    optimizer = args.optimizer
    loss_function = args.loss_function
    scheduler = args.scheduler
    train_dataprovider = args.train_dataprovider
    
    t1 = time.time()
    Top1_err, Top5_err = 0.0, 0.0
    model.train()
    
    trained_group = args.evo_controller.trained_group
    
    for iters in range(1, val_interval + 1):
        scheduler.step()
        if bn_process:
            adjust_bn_momentum(model, iters)
        
        structure_father = random.choice(trained_group)
        
        all_iters += 1
        d_st = time.time()
        data, target = train_dataprovider.next()
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        
        data_time = time.time() - d_st
        
        
        
        structure = structure_father.structure
        
        output = model(data,structure)
        loss = loss_function(output, target)
        
        optimizer.zero_grad()
        
        
        loss.backward()
        optimizer.step()
        
        
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        Top1_err += (1 - prec1.item() / 100)
        Top5_err += (1 - prec5.item() / 100)

        if all_iters % args.display_interval == 0:
            printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(all_iters, scheduler.get_lr()[0], loss.item()) + \
                        'Top-1 err = {:.6f},\t'.format(Top1_err / args.display_interval) + \
                        'Top-5 err = {:.6f},\t'.format(Top5_err / args.display_interval) + \
                        'data_time = {:.6f},\ttrain_time = {:.6f}'.format(data_time, (time.time() - t1) / args.display_interval)
            logging.info(printInfo)
            t1 = time.time()
            Top1_err, Top5_err = 0.0, 0.0

    return all_iters

def validate(model, device, args, *, all_iters=None):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    loss_function = args.loss_function
    val_dataprovider = args.val_dataprovider
    trained_group = args.evo_controller.trained_group
        
#     max_val_iters = args.val_times
    t1  = time.time()
    with torch.no_grad():
        for i in range(len(trained_group)):
            data, target = val_dataprovider.next()
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)
            
            structure_father = trained_group[i]
            structure = structure_father.structure
            
            output = model(data,structure)
            loss = loss_function(output, target)
            
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            if structure_father.count == 0:
                structure_father.loss = float(loss.item()) 
            else:
                structure_father.loss = (float(loss.item())) * (1-args.evo_controller.momentum) + structure_father.loss * args.evo_controller.momentum
            structure_father.count += 1
            
    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(all_iters, objs.avg) + \
              'Top-1 err = {:.6f},\t'.format(1 - top1.avg / 100) + \
              'Top-5 err = {:.6f},\t'.format(1 - top5.avg / 100) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    logging.info(logInfo)


if __name__ == "__main__":
    main()
