import os

import sys
import random
import string
import ast
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

alpha_Y_max_vocab_size=55000

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def try_eval(x):
    try:
        x = ast.literal_eval(x)
        return x
    except:
        return x

def save_as_df(data, col_names, fname):
    pd.DataFrame(data, columns=col_names).to_csv( os.path.join(bolt.ARTIFACT_DIR, f"{fname}.csv"))

def set_alphabets():
    return dict(alpha_X='', alpha_Y=[chr(i) for i in range(sys.maxunicode)][:alpha_Y_max_vocab_size], alpha_W=list('#$%&()*+,-./:;<=>?!@[]^_{|}~')) 