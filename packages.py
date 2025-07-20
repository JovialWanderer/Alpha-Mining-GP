import math
import time
import re
from multiprocessing import Pool
from functools import partial
import numpy as np
import pandas as pd
import pickle
import copy
import os
import random
import bisect
import warnings
from collections import deque
warnings.filterwarnings('ignore')
#-----------Similarity Calculator-----------#
from scipy.stats import pearsonr,ks_2samp
#----------------Volatility Modelling--------#
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
#---------------Plotting--------------------#
import matplotlib.pyplot as plt
import seaborn as sns
#---------------Backtest--------------------#
import vectorbt as vbt

#--------------Ablation Markers-------------#
from dtw import dtw
#--------------SEED for reproducability-------------#
SEED = 119
random.seed(SEED)
np.random.seed(SEED)

# Create a global random state object
global_random_state = random.Random(SEED)
global_np_random_state=np.random.RandomState(SEED)