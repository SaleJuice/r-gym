# _*_ coding: utf-8 _*_
# @File        : dqn_demo.py
# @Time        : 2021/10/23 19:16
# @Author      : SaleJuice
# @E-Mail      : linxzh@shanghaitech.edu.cn
# @Institution : LIMA Lab, ShanghaiTech University, China
# @SoftWare    : PyCharm


import time
import math
import random
import pprint
import numpy as np
import gym
import rgym
from gym import spaces, logger
from gym.utils import seeding

import torch
from tianshou.utils.net.common import Net
from tianshou.data import Collector, ReplayBuffer, PrioritizedReplayBuffer
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from trainer import my_trainer


def sim():
    # env √
    print("env")
    env = gym.make("CartPole-v0")
    env.spec.reward_threshold = 250
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    # model √
    print("model")
    net = Net(state_shape, action_shape, hidden_sizes=[128, 128, 128], device='cpu').to('cpu')
    # optim = torch.optim.Adam(net.parameters(), lr=1e-1)
    optim = torch.optim.RMSprop(net.parameters(), lr=1e-2)

    # policy √
    print("policy")
    policy = DQNPolicy(model=net, optim=optim, discount_factor=0.9, estimation_step=20, target_update_freq=10, is_double=True)
    # policy.load_state_dict(torch.load("policy.pth"))

    # buffer √
    print("buffer")
    buf = PrioritizedReplayBuffer(size=100000, alpha=0.7, beta=0.5)

    # collector √
    print("collector")
    train_collector = Collector(policy=policy, env=env, buffer=buf, exploration_noise=True)
    test_collector = Collector(policy=policy, env=env, exploration_noise=True)

    # # prepare √
    # print("prepare")
    # result = test_collector.collect(n_step=2000, render=True, random=True)
    # pprint.pprint(result)
    # exit()

    # train_fn
    print("train_fn")
    def train_fn(epoch, env_step):
        if epoch <= 10:
            policy.set_eps(0.25)
        elif epoch <= 50:
            policy.set_eps(0.2)
        elif epoch <= 100:
            policy.set_eps(0.15)
        elif epoch <= 250:
            policy.set_eps(0.1)
        else:
            policy.set_eps(0.09)

    def test_fn(epoch, env_step):
        policy.set_eps(0)

    def stop_fn(mean_rewards):
        return mean_rewards >= 1000

    def save_fn(policy):
        torch.save(policy.state_dict(), 'policy.pth')

    # trainer
    print("training")
    result = my_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=None,
        test_in_train=False,
        max_epoch=1000,
        step_per_epoch=500,
        step_per_collect=10,
        episode_per_test=1,
        batch_size=128,
        update_per_step=3,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_fn=save_fn
    )


if __name__ == '__main__':
    print("Start!")
    sim()
    print("Finish!")
