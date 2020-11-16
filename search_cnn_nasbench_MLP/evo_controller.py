import random
import copy
import torch
from torch import nn
import logging
from utils import AvgrageMeter

class case():
    def __init__(self,structure):
        self.structure = structure
        self.loss = 15
        self.count = 0
class evolutionary():
    def __init__(self, max_population = 64,select_number = 32, mutation_len = 4,mutation_number = 1,p_opwise = 0.5,momentum = 0.):
        self.number = max_population
        self.max = max_population
        self.mutation_len = mutation_len
        self.mutation_number = mutation_number
        self.select_number = select_number
        self.momentum = momentum
        self.group = []
        self.p_opwise = p_opwise
        self.elite = []
        
        self.MLP = nn.Sequential(
            nn.Linear(6, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        ).cuda()
        
        self.learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.MLP.parameters(), lr=self.learning_rate)
        
        
        for i in range(self.number):
            structure = self.get_structure()
            self.group.append(case(structure))

        self.trained_group = random.sample(self.group,self.select_number)
        self.criterion = torch.nn.MSELoss().cuda()
        
    def get_structure(self):
        structure = []
        for i in range(3):
            node = []
            for k in range(i+1):
                node.append(random.choice([0,1,2,3,4]))
            structure.append(node)
        return structure
    def mutation(self,parent):
        kid = case(copy.deepcopy(parent.structure))
        for k in range(3):
            for i in range(k+1):
                if random.random()>(1-self.p_opwise):
                    kid.structure[k][i] = random.choice([0,1,2,3,4])
                else:
                    continue
        return kid
    
    def update_MLP(self):
        all_archs = torch.zeros(self.max,6).cuda()
        all_target = torch.zeros(self.max).cuda()
        self.MLP.train()
        for i,structure_father in enumerate(self.group):
            all_archs[i][:] = torch.tensor([item for sublist in structure_father.structure for item in sublist])[:]
            all_target[i] = structure_father.loss
        
        indx = all_target < 15
        all_archs = all_archs[indx,:]
        all_target = all_target[indx]
        epoch = 20
        objs = AvgrageMeter()
        batch_size = 32
        
        
        
        for i in range(epoch):
            start = (batch_size*i)%all_archs.size(0)
            end = start + batch_size
            archs = all_archs[start:end]
            target = all_target[start:end]
            output = self.MLP(archs)
            loss = self.criterion(output,target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            n = archs.size(0)
            objs.update(loss.item(), n)
            
            
        logInfo = 'MLP: loss = {:.6f},\t'.format(objs.avg)
        logging.info(logInfo)
    def inference_MLP(self):
        self.MLP.eval()
        explore_archs = torch.zeros(self.max,6).cuda()
        structures = []
        for i in range(self.max):
            structure = self.get_structure()
            structures.append(structure)
            explore_archs[i][:] = torch.tensor([item for sublist in structure for item in sublist])[:]
        output = self.MLP(explore_archs)
        best_indx = int(output.argmin())
        return structures[best_indx]
    
    def select(self):
        self.update_MLP()
        
#         sorted_group = sorted(self.trained_group,key=lambda x: \
#                               x.loss if x.count > 1 else 100, reverse=False)
        
#         self.elite.append( case(copy.deepcopy(sorted_group[0].structure)) )
        
#         for group in sorted_group:
#             print(group.loss ,group.count)
        
    
        for i in range(self.mutation_len):
            for k in range(self.mutation_number):
                new_structure = self.inference_MLP()
                kid = case(new_structure)
                self.group.append(kid)
        
        if len(self.group) > self.max:
            new_start = len(self.group) - self.max
            self.group = self.group[new_start:]
        
        
        self.trained_group = random.sample(self.group,self.select_number)
        
    def final_stage(self):
        new_start = len(self.elite) - self.max
        
        self.group = self.elite[new_start:]