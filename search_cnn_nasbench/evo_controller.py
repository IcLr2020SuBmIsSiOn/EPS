import random
import copy

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
        
        for i in range(self.number):
            structure = self.get_structure()
            self.group.append(case(structure))

        self.trained_group = random.sample(self.group,self.select_number)
    
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
    
    def select(self):
        sorted_group = sorted(self.trained_group,key=lambda x: \
                              x.loss if x.count > 1 else 100, reverse=False)
        
        self.elite.append( case(copy.deepcopy(sorted_group[0].structure)) )
        
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
        
    def final_stage(self):
        new_start = len(self.elite) - self.max
        
        self.group = self.elite[new_start:]