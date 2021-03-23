import os
import random
import numpy as np
import torch


def set_seed(seed=0):
    """
    set random seed

    Args:
        seed (int, optional): random seed. Defaults to 0.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
