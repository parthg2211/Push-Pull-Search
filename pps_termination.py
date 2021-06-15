from collections import deque
import numpy as np

from pymoo.model.termination import Termination

class PushPullSearchTermination(Termination):
    def __init__(self):
        super().__init__()

        self.z_queue = deque()
        self.n_queue = deque()
    
    def _do_continue(self,algorithm):
        F = algorithm.pop.get("F").astype(float, copy=False)
        epsilon =  1e-3

        #set ideal and nadir points
        z,n = SetIdealNadirPoints(F)
        if len(self.z_queue) < 10 and len(self.n_queue) < 10:
            self.z_queue.append(z)
            self.n_queue.append(n)
        else:
            zkl = self.z_queue.popleft()
            nkl = self.n_queue.popleft()
            self.z_queue.append(z)
            self.n_queue.append(n)

            rk = CalcMaxRateChange(z,zkl,n,nkl)

            if rk < epsilon:
                return False
                
        return True



def SetIdealNadirPoints(F):
    
    m = F.shape[1]
    # intitialize z = ideal point, n as nadir point
    z = np.full((m,1),1e30)
    n = np.full((m,1),-1e30)

    
    N = F.shape[0]
    for i in range(N):
        for j in range(m):
            
            if F[i,j] < z[j,0] :
                z[j,0] = F[i,j]
            
            if F[i,j] > n[j,0] :
                n[j,0] = F[i,j]
    
    return z,n

def CalcMaxRateChange(zk,zkl,nk,nkl):
    
    delta = -1e6
    rzk = -1e30
    rnk = -1e30
    
    for i in range(zk.shape[0]):
        
        x = abs(zk[i,0]-zkl[i,0])/max(zkl[i,0],delta)
        
        if x > rzk:
            rzk = x
        
        y = abs(nk[i,0]-nkl[i,0])/max(nkl[i,0],delta)
        
        if y > rnk:
            rnk = y
    
    rk = max(rzk,rnk)

    return rk