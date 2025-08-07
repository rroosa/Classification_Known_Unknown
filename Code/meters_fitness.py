import torch
import numpy as np 
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error
#--------------------------------------------------------------------------------------------------------------

def compute_fitness(P):
    U = np.ones_like(P) / len(P)
    p_extremo = np.zeros(len(P),dtype=float)
    p_extremo[0] = 1.0
    print(f"Uniform distribution {U}, Assymetric distribution {p_extremo}")

    jsd_value_max = jensenshannon(p_extremo,U, base=2 )**2 
   
    jsd_value =  jensenshannon(P,U, base=2 )**2 

    jsd_norm = jsd_value / jsd_value_max 

    fitness = 1 - jsd_norm 
    #fitness = 1 - jsd_value 
    print(f"Fitness score: {fitness}")
    

    return fitness
