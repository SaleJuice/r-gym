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


if __name__ == '__main__':
    print("Program Start!")
    env = rgym.envs.sim.cartpole_balance.Env()
    try:
        env.reset()
        while True:
            # print(env.step())
            # print(np.clip(np.random.normal(loc=0, scale=0.015/3), )
            # print(np.random.normal(loc=0, scale=0.015/3))
            env.render()
    except:
        env.close()
        print("Program Exit!")