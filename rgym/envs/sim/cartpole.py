# _*_ coding: utf-8 _*_
# @File        : cartpole.py
# @Time        : 2021/12/7 10:16
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

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        self.g = 9.8  # N/kg
        self.m_c = 1.0  # kg
        self.m_p = 0.1  # kg
        self.m_all = (self.m_p + self.m_c)  # kg
        self.l = 0.5  # actually half the pole's length  # m
        self.m_p_l = (self.m_p * self.l)  # kg*m
        self.f_max = 10.0  # N
        self.dt = 0.02  # s

        # Angle at which to fail the episode
        self.angle_threshold = 12 * 2 * math.pi / 360  # rad
        self.position_threshold = 2.4  # m

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.position_threshold * 2,
                np.finfo(np.float32).max,
                self.angle_threshold * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        # kinematic parameters
        self.x = 0
        self.x_dot = 0
        self.theta = 0
        self.theta_dot = 0

        # state parameters
        self.position_pre = 0
        self.position_cur = 0
        self.position_delta = 0

        self.angle_pre = 0
        self.angle_cur = 0
        self.angle_delta = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def kinematics(self, force, model="euler"):
        # kinematic model reference with https://coneural.org/florian/papers/05_cart_pole.pdf
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)

        temp = (force + self.m_p_l * self.theta_dot ** 2 * sin_theta) / self.m_all
        theta_acc = (self.g * sin_theta - cos_theta * temp) / (self.l * (4.0 / 3.0 - self.m_p * cos_theta ** 2 / self.m_all))
        x_acc = temp - self.m_p_l * theta_acc * cos_theta / self.m_all

        if model == "euler":
            self.x = self.x + self.dt * self.x_dot
            self.x_dot = self.x_dot + self.dt * x_acc
            self.theta = self.theta + self.dt * self.theta_dot
            self.theta_dot = self.theta_dot + self.dt * theta_acc
        else:  # semi-implicit euler
            self.x_dot = self.x_dot + self.dt * x_acc
            self.x = self.x + self.dt * self.x_dot
            self.theta_dot = self.theta_dot + self.dt * theta_acc
            self.theta = self.theta + self.dt * self.theta_dot

    def step(self, action=None):
        if action != None:
            assert self.action_space.contains(action), f"action = {action} ({type(action)}) is invalid"

        if action == 1:
            force = self.f_max
        elif action == 0:
            force = -self.f_max
        else:
            force = 0

        self.kinematics(force)

        self.position_cur = self.x + np.random.normal(loc=0, scale=0.0025)
        self.angle_cur = self.theta + np.random.normal(loc=0, scale=0.0025)
        self.position_delta = self.position_cur - self.position_pre
        self.angle_delta = self.angle_cur - self.angle_pre
        self.position_pre = self.position_cur
        self.angle_pre = self.angle_cur

        self.state = (self.position_cur, self.position_delta, self.angle_cur, self.angle_delta)

        done = bool(
            self.position_cur < -self.position_threshold
            or self.position_cur > self.position_threshold
            or self.angle_cur < -self.angle_threshold
            or self.angle_cur > self.angle_threshold
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
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

        return np.array(self.state), reward, done, {}

    def reset(self):
        (self.x, self.x_dot, self.theta, self.theta_dot) = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))

        self.position_pre = self.x
        self.angle_pre = self.theta
        self.position_cur = self.x
        self.angle_cur = self.theta
        self.position_delta = self.position_cur - self.position_pre
        self.angle_delta = self.angle_cur - self.angle_pre

        self.state = (self.position_cur, self.position_delta, self.angle_cur, self.angle_delta)

        self.steps_beyond_done = None

        return np.array(self.state)

    def render(self, mode="human"):
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
        action = random.randint(0, 1)
        while True:
            env.render()
            observation, reward, done, _ = env.step(action)
            print(observation)
            if observation[0] > 3.5:
                action = 0
            elif observation[0] < -3.5:
                action = 1
    except:
        env.close()
        print("Program Exit!")
