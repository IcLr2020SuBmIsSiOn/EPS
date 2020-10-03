import random
import copy
from genotypes import *

class case():
    def __init__(self,structure):
        self.structure = structure
        self.loss = 0
        self.count = 0
        self.last_pos = 0
class evolutionary():
    def __init__(self, max_population = 64,select_number = 32, mutation_len = 4,mutation_number = 1):
        self.number = max_population
        self.max = max_population
        self.mutation_len = mutation_len
        self.mutation_number = mutation_number
        self.select_number = select_number
        self.momentum = 0.0
        self.group = []
        
        for i in range(self.number):
            structure = self.get_structure()
            self.group.append(case(structure))

        self.trained_group = random.sample(self.group,self.select_number)
        
    
    def get_structure(self):
        structure = []
        indexess = []
        for i in range(0,8):
            indexes = []
            for k in range(0,i+1):
                indexes.append(0)
            indexess.append(indexes)
        for i in range(0,8):
            idd = random.sample(range(i+1),1)

            indexess[i][idd[0]] = random.choice([1,2,3,4])


        structure.append(indexess)
        return structure
    
    def mutation(self,parent):
        kid = case(copy.deepcopy(parent.structure))
        
        p_edgewise = 0.1
        p_opwise = 0.2

        for indexess in kid.structure:
            for i in range(0,8):
                if_none = []
                if_ops = []
                for k in range(i+1):
                    if indexess[i][k]>0:
                        if random.random()>1 - p_opwise:
                            indexess[i][k] = random.choice([1,2,3,4])
                        if_ops.append(k)
                    else:
                        if_none.append(k)            
                if i > 0 and random.random()>1 - p_edgewise:
                    s1 = random.choice(if_ops)
                    s2 = random.choice(if_none)
                    indexess[i][s2] = indexess[i][s1]
                    indexess[i][s1] = 0
        return kid
    
    def select(self):
        
        sorted_group = sorted(self.trained_group,key=lambda x: \
                              x.loss if x.count >= 1 else 10000, reverse=False)
                
        
        for group in sorted_group:
            print(group.loss ,group.count)
        
        for i in range(self.mutation_len):
            for k in range(self.mutation_number):
                kid = self.mutation(sorted_group[i])
                self.group.append(kid)
        
        if len(self.group) > self.max:
            new_start = len(self.group) - self.max
            self.group = self.group[new_start:]
        
        
        self.trained_group = random.sample(self.group,self.select_number)
    def indx2genotype(self,indx,search_space = PRIMITIVES):
        recurrent = []

        for i in range(0,8):
            for k in range(i+1):
                if indx[0][i][k] > 0:
                    recurrent.append((search_space[indx[0][i][k]],k))


        return Genotype(recurrent = recurrent,concat=range(1, 9))

    