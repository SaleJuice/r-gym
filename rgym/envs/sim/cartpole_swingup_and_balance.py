# _*_ coding: utf-8 _*_
# @File        : cartpole_swingup_and_balance.py
# @Time        : 2021/12/6 16:45
# @Author      : SaleJuice
# @E-Mail      : linxzh@shanghaitech.edu.cn
# @Institution : LIMA Lab, ShanghaiTech University, China
# @SoftWare    : PyCharm


import time
import math
import random
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding


class Env(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self):
        # kinematic parameters
        self.g = 9.82  # N/kg
        self.m_c = 0.5  # kg
        self.m_p = 0.5  # kg
        self.m_all = (self.m_p + self.m_c)  # kg
        self.l = 0.5  # actually half the pole's length  # m
        self.m_p_l = (self.m_p * self.l)  # kg*m
        self.f_max = 10.0  # N
        self.dt = 0.02  # s
        self.fri = 0  # friction coefficient

        self.x = 0
        self.x_dot = 0
        self.theta = 0
        self.theta_dot = 0

        # state parameters
        self.t = 0  # timestep
        self.t_limit = 100

        self.position_pre = 0
        self.position_cur = 0
        self.position_delta = 0
        self.position_threshold = 2.4  # m

        self.angle_pre = 0
        self.angle_cur = 0
        self.angle_delta = 0
        self.angle_threshold = 12 * 2 * math.pi / 360  # rad

        high = np.array(
            [
                self.position_threshold,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # TODO: gaussian noise part.
        # gaussian noise

        # others
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def kinematic_fn(self, force):
        sin_theta = math.sin(self.theta)
        cos_theta = math.cos(self.theta)

        # x_acc = (-2 * self.m_p_l * (self.theta_dot ** 2) * sin_theta + 3 * self.m_p * self.g * sin_theta * cos_theta + 4 * force - 4 * self.fri * self.x_dot)\
        #               / (4 * self.m_all - 3 * self.m_p * cos_theta ** 2)
        # theta_acc = (-3 * self.m_p_l * (self.theta_dot ** 2) * sin_theta * cos_theta + 6 * self.m_all * self.g * sin_theta + 6 * (force - self.fri * self.x_dot) * cos_theta)\
        #                   / (4 * self.l * self.m_all - 3 * self.m_p_l * cos_theta ** 2)

        temp = (force + self.m_p_l * self.theta_dot ** 2 * sin_theta) / self.m_all
        theta_acc = (self.g * sin_theta - cos_theta * temp) / (self.l * (4.0 / 3.0 - self.m_p * cos_theta ** 2 / self.m_all))
        x_acc = temp - self.m_p_l * theta_acc * cos_theta / self.m_all

        self.x = self.x + self.x_dot * self.dt
        self.x_dot = self.x_dot + x_acc * self.dt
        self.theta = self.theta + self.theta_dot * self.dt
        self.theta_dot = self.theta_dot + theta_acc * self.dt

    def reward_fn(self, state, done):
        # TODO: new reward function
        goal = np.array([0.0, self.l * 2])
        pole_x = self.l * 2 * np.sin(state[2])
        pole_y = self.l * 2 * np.cos(state[2])
        position = np.array([state[0] + pole_x, pole_y])
        squared_distance = np.sum((position - goal) ** 2)
        squared_sigma = 0.6 ** 2
        # cost = 1 - np.exp(-0.5 * squared_distance / squared_sigma)
        cost = 1 / (squared_distance + 1)

        if not done:
            reward = cost
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = cost
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return reward

    def step(self, action=None, freq=None):
        if action != None:
            force = np.clip(action, -1.0, 1.0)[0] * self.f_max
        else:
            force = 0

        self.kinematic_fn(force)

        self.position_cur = self.x  # + np.random.normal(loc=0, scale=0.001)
        self.angle_cur = self.theta  # + np.random.normal(loc=0, scale=0.001)
        self.position_delta = self.position_cur - self.position_pre
        self.angle_delta = self.angle_cur - self.angle_pre
        self.position_pre = self.position_cur
        self.angle_pre = self.angle_cur

        self.state = (self.position_cur, self.position_delta, self.angle_cur, self.angle_delta)

        self.t += 1

        done = bool(
            self.position_cur < -self.position_threshold
            or self.position_cur > self.position_threshold
            or self.t >= self.t_limit
        )

        reward = self.reward_fn(self.state, done)

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.x = self.np_random.uniform(low=-0.05, high=0.05)
        self.theta = self.np_random.uniform(low=3.1, high=3.2)

        self.position_pre = self.x
        self.angle_pre = self.theta
        self.position_cur = self.x
        self.angle_cur = self.theta
        self.position_delta = self.position_cur - self.position_pre
        self.angle_delta = self.angle_cur - self.angle_pre

        self.state = (self.position_cur, self.position_delta, self.angle_cur, self.angle_delta)

        self.t = 0  # timestep
        self.steps_beyond_done = None

        return np.array(self.state)

    def render(self, mode='human', close=False):
        screen_width = 600
        screen_height = 400

        world_width = self.position_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.l)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    print("Program Start!")
    env = Env()
    try:
        env.reset()
        while True:
            print(env.step([0.1]))
            env.render()
    except:
        env.close()
        print("Program Exit!")