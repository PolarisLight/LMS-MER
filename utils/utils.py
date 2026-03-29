import random
import numpy as np
import torch
import os

def set_seed(seed_value):
    # Python RNG
    random.seed(seed_value)
    
    # NumPy RNG
    np.random.seed(seed_value)
    
    # PyTorch CPU RNG
    torch.manual_seed(seed_value)
    
    # PyTorch CUDA RNG
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    
    # Deterministic CuDNN behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Hash seed for Python process
    os.environ['PYTHONHASHSEED'] = str(seed_value)