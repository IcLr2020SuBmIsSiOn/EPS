import sys
import os
import shutil
import logging
import inspect
import pickle
import argparse
import numpy as np
from darts_wrapper_discrete import DartsWrapper
import random
import pickle
import json
import copy
from genotypes import *
import time
from evo_controller import evolutionary
    
def get_args():
    parser = argparse.ArgumentParser("EVOVIN")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval-resume', type=str, default='./snet_detnas.pkl', help='path for eval model')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--total-iters', type=int, default=80000, help='total iters')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--label-smooth', type=float, default=0.1, help='label smoothing')

    parser.add_argument('--auto-continue', type=bool, default=False, help='report frequency')
    parser.add_argument('--display-interval', type=int, default=10, help='report frequency')
    parser.add_argument('--val-interval', type=int, default=200, help='report frequency')
    parser.add_argument('--val-times', type=int, default=1, help='report frequency')

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
    
    logging.info(args)
    
    
    
    
    all_iters = 0
    
    
    
    args.evo_controller = evolutionary(args.max_population,args.select_number, args.mutation_len,args.mutation_number)
    
    
    path = "./record"+"_"+str(args.max_population)+"_"+str(args.select_number)+"_"+str(args.mutation_len)+"_"+str(args.mutation_number)+ "_" + str(args.val_interval) + "_"+str(args.rand_seed)
    
    
    
#     args.evo_controller.trained_group = args.evo_controller.group
    
    model = DartsWrapper('./models', args.rand_seed, args.val_interval,args.val_times,args.evo_controller,batch_size = args.batch_size)
    
    while all_iters < args.total_iters:
        
        arch = args.evo_controller.indx2genotype(random.choice(args.evo_controller.trained_group).structure)
        
        
        model.train_batch(arch)
        
        
        
        if all_iters > 1 and all_iters%args.val_interval == 0:
            results = []
            for structure_father in args.evo_controller.group:
                results.append([structure_father.structure,structure_father.loss,structure_father.count,structure_father.last_pos])
                
            if not os.path.exists(path):
                os.mkdir(path)
                
            with open(path + '/%06d-ep.txt'%all_iters,'w') as tt:
                json.dump(results,tt)
            
            args.evo_controller.select()


        
        all_iters = all_iters + 1
        
        
    ###end
    results = []
    for structure_father in args.evo_controller.group:
        results.append([structure_father.structure,structure_father.loss,structure_father.count])
    with open(path + '/%06d-ep.txt'%all_iters,'w') as tt:
        json.dump(results,tt)
    
    

if __name__ == "__main__":
    main()