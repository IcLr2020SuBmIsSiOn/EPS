import random
from genotypes import *
import copy
import numpy as np

class case():
    def __init__(self,structure,size = 0):
        self.structure = structure
        self.loss = 0
        self.count = 0
        self.size = size
class evolutionary():
    def __init__(self, max_population = 64,select_number = 32, mutation_len = 4,mutation_number = 1,p_edgewise = 0.2,p_opwise = 0.2):
        self.number = max_population
        self.max = max_population
        self.mutation_len = mutation_len
        self.mutation_number = mutation_number
        self.select_number = select_number
        self.momentum = 0.
        self.group = []
        self.p_edgewise = p_edgewise
        self.p_opwise = p_opwise
        self.elite = []
        self.ops_36_20 = [[[0.0,
   0.0,
   0.0,
   0.02635199999999993,
   0.056591999999999976,
   0.063504,
   0.0282960000000001,
   0.031752],
  [0.0,
   0.0,
   0.0,
   0.02635199999999993,
   0.056591999999999976,
   0.063504,
   0.0282960000000001,
   0.031752],
  [0.0,
   0.0,
   0.0,
   0.02635199999999993,
   0.056591999999999976,
   0.063504,
   0.0282960000000001,
   0.031752],
  [0.0,
   0.0,
   0.0,
   0.02635199999999993,
   0.056591999999999976,
   0.063504,
   0.0282960000000001,
   0.031752],
  [0.0,
   0.0,
   0.0,
   0.0,
   0.056591999999999976,
   0.063504,
   0.0282960000000001,
   0.031752],
  [0.0,
   0.0,
   0.0,
   0.02635199999999993,
   0.056591999999999976,
   0.063504,
   0.0282960000000001,
   0.031752],
  [0.0,
   0.0,
   0.0,
   0.02635199999999993,
   0.056591999999999976,
   0.063504,
   0.0282960000000001,
   0.031752],
  [0.0,
   0.0,
   0.0,
   0.0,
   0.056591999999999976,
   0.063504,
   0.0282960000000001,
   0.031752],
  [0.0,
   0.0,
   0.0,
   0.0,
   0.056591999999999976,
   0.063504,
   0.0282960000000001,
   0.031752],
  [0.0,
   0.0,
   0.0,
   0.02635199999999993,
   0.056591999999999976,
   0.063504,
   0.0282960000000001,
   0.031752],
  [0.0,
   0.0,
   0.0,
   0.02635199999999993,
   0.056591999999999976,
   0.063504,
   0.0282960000000001,
   0.031752],
  [0.0,
   0.0,
   0.0,
   0.0,
   0.056591999999999976,
   0.063504,
   0.0282960000000001,
   0.031752],
  [0.0,
   0.0,
   0.0,
   0.0,
   0.056591999999999976,
   0.063504,
   0.0282960000000001,
   0.031752],
  [0.0,
   0.0,
   0.0,
   0.0,
   0.056591999999999976,
   0.063504,
   0.0282960000000001,
   0.031752]],
 [[0.0,
   0.0,
   0.0,
   0.0,
   0.35985599999999995,
   0.40823999999999994,
   0.1799280000000001,
   0.20412000000000008],
  [0.0,
   0.0,
   0.0,
   0.0,
   0.35985599999999995,
   0.40823999999999994,
   0.1799280000000001,
   0.20412000000000008],
  [0.0,
   0.0,
   0.0,
   0.0,
   0.35985599999999995,
   0.40823999999999994,
   0.1799280000000001,
   0.20412000000000008],
  [0.0,
   0.0,
   0.0,
   0.0,
   0.35985599999999995,
   0.40823999999999994,
   0.1799280000000001,
   0.20412000000000008],
  [0.0,
   0.0,
   0.0,
   0.0,
   0.35985599999999995,
   0.40823999999999994,
   0.1799280000000001,
   0.20412000000000008],
  [0.0,
   0.0,
   0.0,
   0.0,
   0.35985599999999995,
   0.40823999999999994,
   0.1799280000000001,
   0.20412000000000008],
  [0.0,
   0.0,
   0.0,
   0.0,
   0.35985599999999995,
   0.40823999999999994,
   0.1799280000000001,
   0.20412000000000008],
  [0.0,
   0.0,
   0.0,
   0.0,
   0.35985599999999995,
   0.40823999999999994,
   0.1799280000000001,
   0.20412000000000008],
  [0.0,
   0.0,
   0.0,
   0.0,
   0.35985599999999995,
   0.40823999999999994,
   0.1799280000000001,
   0.20412000000000008],
  [0.0,
   0.0,
   0.0,
   0.0,
   0.35985599999999995,
   0.40823999999999994,
   0.1799280000000001,
   0.20412000000000008],
  [0.0,
   0.0,
   0.0,
   0.0,
   0.35985599999999995,
   0.40823999999999994,
   0.1799280000000001,
   0.20412000000000008],
  [0.0,
   0.0,
   0.0,
   0.0,
   0.35985599999999995,
   0.40823999999999994,
   0.1799280000000001,
   0.20412000000000008],
  [0.0,
   0.0,
   0.0,
   0.0,
   0.35985599999999995,
   0.40823999999999994,
   0.1799280000000001,
   0.20412000000000008],
  [0.0,
   0.0,
   0.0,
   0.0,
   0.35985599999999995,
   0.40823999999999994,
   0.1799280000000001,
   0.20412000000000008]]]
#[[0,0,0,0,0.056591999999999976,0.063504,0.0282960000000001,0.031752],[0,0,0,0,0.35985599999999995,0.40823999999999994,0.1799280000000001,0.20412000000000008]]
        
        self.params_min = 1.370134
        self.params_max = 5.144086
        
        for i in range(self.number):
            structure = self.get_structure()
            self.group.append(case(structure,self.get_params(structure)))

        self.trained_group = random.sample(self.group,self.select_number)
    
    def get_params(self, structure):
        params_size = self.params_min
        for i in range(2):
            step = 0
            for k in range(0,4):
                for indx in structure[i][k]:
                    params_size += self.ops_36_20[i][step][indx]
                    step+=1
        return params_size
    
    def get_structure(self):
        structure = []
        for i in range(2):
            indexess = []
            for i in range(0,4):
                indexes = []
                for k in range(0,2+i):
                    indexes.append(0)
                indexess.append(indexes)
            for i in range(0,4):
                idd = random.sample(range(i+2),2)
                indexess[i][idd[0]] = random.choice([1,2])
                indexess[i][idd[1]] = random.choice([1,2])        
            structure.append(indexess)
        return structure
    def mutation(self,parent):
        kid = case(copy.deepcopy(parent.structure))
        
        p_edgewise = self.p_edgewise
        p_opwise = self.p_opwise
        
        for indexess in kid.structure:
            for i in range(0,4):
                if_none = []
                if_ops = []
                for k in range(i+2):
                    if indexess[i][k]>0:
                        if random.random()>(1-p_opwise):
                            indexess[i][k] = random.choice([1,2])
                        if_ops.append(k)
                    else:
                        if_none.append(k)            
                if i > 0 and random.random()>(1-p_edgewise):
                    s1 = random.choice(if_ops)
                    s2 = random.choice(if_none)
                    indexess[i][s2] = random.choice([1,2])#indexess[i][s1]
                    indexess[i][s1] = 0

        kid.size = self.get_params(kid.structure)
        return kid
    
    def select(self):
#         sorted_group = sorted(self.group,key=lambda x: \
#                               x.loss/x.count if x.count>1 else 100, reverse=False)
#         for group in sorted_group:
#             print(group.loss/group.count if group.count>1 else 100)

        sorted_group = sorted(self.trained_group,key=lambda x: \
                              x.loss  if x.count >= 1 else 100, reverse=False)
        
#         self.elite.append( case(copy.deepcopy(sorted_group[0].structure)) )
        
        
        for group in sorted_group:
#             print(group.loss ,group.count)
            print(group.loss ,group.count)
        
        for i in range(self.mutation_len):
            for k in range(self.mutation_number):
                kid = self.mutation(sorted_group[i])
                self.group.append(kid)
        
        if len(self.group) > self.max:
            new_start = len(self.group) - self.max
            self.group = self.group[new_start:]
        
        
        self.trained_group = random.sample(self.group,self.select_number)
        
#     def final_stage(self):
#         new_start = len(self.elite) - self.max
        
#         self.group = self.elite[new_start:]
        
  
