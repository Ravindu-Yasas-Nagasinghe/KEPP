import glob
import os
import random
import time
from collections import OrderedDict

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.distributed import ReduceOp
import torch.nn.functional as F
from dataloader.data_load_mlp import PlanningDataset
from model.helpers import get_lr_schedule_with_warmup, Logger
import torch.nn as nn
from utils import *
from logging import log
from utils.args import get_args
import numpy as np

args = get_args()
os.environ['PYTHONHASHSEED'] = str(args.seed)
train_dataset = PlanningDataset(
            args.root,
            args=args,
            is_val=False,
            model=None,
        )