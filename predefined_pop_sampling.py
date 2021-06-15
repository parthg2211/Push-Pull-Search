import numpy as np

from pymoo.model.sampling import Sampling
from pymoo.model.population import Population

class PredefinedPopulation(Sampling):

    def __init__(self,pop = None):
        
        super().__init__()
        if pop is not None:
            self.pop = pop
    
    def _do(self,problem,n_samples,**kwargs):
        if n_samples != self.pop.shape[0]:
            raise ValueError("previous and current population size do not match")

        return self.pop
