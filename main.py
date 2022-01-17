# _*_ coding: utf-8 _*_
# @File        : main.py
# @Time        : 2022/1/10 21:51
# @Author      : SaleJuice
# @E-Mail      : linxzh@shanghaitech.edu.cn
# @Institution : LIMA Lab, ShanghaiTech University, China
# @SoftWare    : PyCharm

import argparse
import os
import pprint

import gym
import rgym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PGPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net

import time
import tkinter as tk
from tkinter import *


if __name__ == '__main__':
    for i in range(100):
        time.sleep(1)
        print(np.random.randint(0, 2))

