# _*_ coding: utf-8 _*_
# @File        : dqn.py
# @Time        : 2022/1/12 13:38
# @Author      : SaleJuice
# @E-Mail      : linxzh@shanghaitech.edu.cn
# @Institution : LIMA Lab, ShanghaiTech University, China
# @SoftWare    : PyCharm

import argparse
import os
import pprint

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net

import rgym
from trainer import op_trainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='CartPole-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps-test', type=float, default=0.0)
    parser.add_argument('--eps-train', type=float, default=0.25)
    parser.add_argument('--buffer-size', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--n-step', type=int, default=10)  # 10
    parser.add_argument('--target-update-freq', type=int, default=100)  # 100
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--step-per-epoch', type=int, default=10000)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=5)  # 3
    parser.add_argument('--batch-size', type=int, default=128)  # 256
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64, 64])  # 128 128 128
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--prioritized-replay', action="store_true", default=True)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def test_dqn(args=get_args()):
    # env = gym.make(args.task)
    env = rgym.envs.real.cartpole_swingup.Env("ttyUSB0", 750)
    env.reset(touch=True)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # you can also use tianshou.env.SubprocVectorEnv
    # train_envs = env
    train_envs = DummyVectorEnv([lambda: env for _ in range(args.training_num)])
    # test_envs = env
    test_envs = DummyVectorEnv([lambda: env for _ in range(args.test_num)])

    # seed
    args.seed = np.random.randint(0, 1000)
    # args.seed = 718
    # args.seed = 131
    args.seed = 91
    print(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    Q_param = V_param = {"hidden_sizes": [128]}

    # model
    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        dueling_param=(Q_param, V_param),
    ).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    # optim = torch.optim.RMSprop(net.parameters(), lr=1e-2)
    policy = DQNPolicy(
        net,
        optim,
        args.gamma,
        args.n_step,
        target_update_freq=args.target_update_freq,
    )
    # policy.load_state_dict(torch.load("policy.pth"))

    # buffer
    if args.prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            alpha=args.alpha,
            beta=args.beta,
        )
    else:
        buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))

    # collector
    train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num, random=True)


    # log
    log_path = os.path.join(args.logdir, args.task, 'dqn')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), f'cart_position_{args.seed}_dqn.pth')

    def stop_fn(mean_rewards):
        return mean_rewards >= 300

    def train_fn(epoch, env_step):
        # eps annnealing, just a demo
        if env_step <= 3000:
            policy.set_eps(0.5)
        elif env_step <= 30000:
            policy.set_eps(args.eps_train)
        elif env_step <= 100000:
            eps = args.eps_train - (env_step - 30000) / \
                70000 * (0.9 * args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * args.eps_train)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # trainer
    result = op_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        update_per_step=args.update_per_step,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_fn=save_fn,
        logger=logger,
        test_in_train=False
    )
    # assert stop_fn(result['best_reward'])

    if __name__ == '__main__':
        pprint.pprint(result)
        print("Let's watch its performance!")
        # env = gym.make(args.task)
        env = rgym.envs.real.cartpole_swingup.Env("ttyUSB0", 1000)
        policy.eval()
        policy.set_eps(args.eps_test)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=3, render=args.render)
        rews, lens, rew_std, len_std = result["rews"], result["lens"], result["rew_std"], result["len_std"]
        print(f"Final reward: {rews.mean()}±{rew_std}, length: {lens.mean()}±{len_std}")


if __name__ == '__main__':
    test_dqn()
