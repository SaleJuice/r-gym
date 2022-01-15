# _*_ coding: utf-8 _*_
# @File        : cartpole_swingup.py
# @Time        : 2021/12/7 10:29
# @Author      : SaleJuice
# @E-Mail      : linxzh@shanghaitech.edu.cn
# @Institution : LIMA Lab, ShanghaiTech University, China
# @SoftWare    : PyCharm

import time
import math
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding

from rgym.utils.easy_serial import MacBackground

import matplotlib.pyplot as plt
import pygame

np.set_printoptions(suppress=True)  # cancel scientific notation output


class Env(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, portx, t_limit=100):
        # connect with real-machine by uart
        self.ser = MacBackground()  # diff system with diff background
        assert (self.ser.connect(portx)), f"\n\tCan't connect real env by '{portx}'!\n\t\tThere only have ports: {self.ser.port_list}"

        # time param
        self.sta_time = 0
        self.end_time = 0

        # offset
        self.__ox = 0
        self.__oa = 0

        # private variables
        self.__px = 0  # previous position of cart
        self.__x = 0  # current position of cart
        self.__dx = 0  # delta position of cart

        self.__pa = 0  # previous angle of pole
        self.__a = 0  # current angle of pole
        self.__da = 0  # delta angle of pole

        self.__k = 0  # current value of push-button
        self.__p = 0  # output wait to send

        # kinematic parameters
        # self.g = 9.8  # N/kg
        # self.m_c = 1.0  # kg
        # self.m_p = 0.1  # kg
        # self.m_all = (self.m_p + self.m_c)  # kg
        # self.l = 0.5  # actually half the pole's length  # m
        # self.m_p_l = (self.m_p * self.l)  # kg*m
        self.f_max = 10.0  # N
        self.dt = 0.02  # s

        self.x = 0
        self.x_dot = 0
        self.theta = 0
        self.theta_dot = 0

        # state parameters
        self.t = 0
        self.t_limit = t_limit
        self.position_pre = 0
        self.position_cur = 0
        self.position_delta = 0
        self.position_threshold = 132 / 2  # cm

        self.angle_pre = 0
        self.angle_cur = 0
        self.angle_delta = 0
        self.angle_threshold = 12 * 2 * math.pi / 360  # rad

        high = np.array(
            [
                self.position_threshold * 2,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # others
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __input(self, type="cur"):
        num, res, _ = self.ser.read("all")
        if num >= 16 and res[res.rfind(')') - 14] == '(':
            res = res[res.rfind(')') - 13:res.rfind(')')]
            raw_data = list(map(int, res.split(',')))
            if type == "cur":
                self.__x = (raw_data[0] - self.__ox)
                self.__a = (raw_data[1] - self.__oa)
            else:  # "pre"
                self.__px = (raw_data[0] - self.__ox)
                self.__pa = (raw_data[1] - self.__oa)
            self.__k = raw_data[2]
            return True
        return False

    def __output(self, t, p):
        t = int(t)
        p = int(p)
        f = int(self.dt / 0.005)  # the frequency of real machine is 200Hz(0.005s)
        if self.ser.write(f"({t},{p},{f})\n"):
            return True
        return False

    def kinematic_fn(self, force):
        self.__p = max(-1800, min(force, 1800))
        self.__output(0, self.__p)

        # ensure the gap of control is stable
        # print(time.perf_counter() - self.sta_time)
        while (self.end_time - self.sta_time) < self.dt:
            self.end_time = time.perf_counter()
        # print(time.perf_counter() - self.sta_time)
        self.sta_time = time.perf_counter()

        self.__input()
        self.__dx = self.__x - self.__px
        self.__px = self.__x
        self.__da = self.__a - self.__pa
        self.__pa = self.__a
        # keep speed continually
        if self.__da > 2048:
            self.__da -= 4096
        elif self.__da < -2048:
            self.__da += 4096

    def reward_fn(self, state, done):
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

    def step(self, action=None):
        if action != None:
            assert self.action_space.contains(action), f"action = {action} ({type(action)}) is invalid"

        if action == 1:
            force = self.f_max
        elif action == 0:
            force = -self.f_max
        else:
            force = 0

        self.kinematic_fn(force)

        self.position_cur = self.__x * (self.position_threshold * 2 / 134200)
        self.position_delta = self.__dx * (self.position_threshold * 2 / 134200)
        self.angle_cur = self.__a * (math.pi * 2 / 4096)
        self.angle_delta = self.__da * (math.pi * 2 / 4096)

        self.state = (self.position_cur, self.position_delta, self.angle_cur, self.angle_delta)
        self.t += 1

        done = bool(
            self.position_cur < -self.position_threshold * 0.7
            or self.position_cur > self.position_threshold * 0.7
            or self.t >= self.t_limit
        )

        # distance
        # goal = np.array([0.0, self.l * 2])
        # pole_x = self.l * 2 * np.sin(self.angle_cur)
        # pole_y = self.l * 2 * np.cos(self.angle_cur)
        # position = np.array([self.position_cur + pole_x, pole_y])
        # squared_distance = np.sum((position - goal) ** 2)  # max(squared_distance) = self.l * 4
        # squared_sigma = 0.5 ** 2
        # cost = np.exp(-0.5 * squared_distance / squared_sigma)
        # # cost = 1 / (squared_distance + 1)

        # angle
        goal = np.array([0, -1])
        pole_x = np.sin(self.angle_cur)
        pole_y = np.cos(self.angle_cur)
        position = np.array([pole_x, pole_y])
        squared_distance = np.sum((position - goal) ** 2)
        squared_sigma = 0.5 ** 2
        cost = np.exp(-0.5 * squared_distance / squared_sigma)

        reward = 0.0
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

        # reward = self.reward_fn(self.state, done)

        return np.array(self.state), reward, done, {}

    def reset(self, touch=False):
        if touch:
            # touch edge
            self.__input()
            while self.__k != 1:
                self.__input()
                self.__output(0, -800)
            self.__ox = self.__x + 67100  # 66569
            self.__output(0, 800)

        # back to the middle
        self.__input()
        randomevent = self.np_random.uniform(low=-4500, high=4500)
        while abs(self.__x - randomevent) >= 100:
            self.__input()
            if abs(self.__x - randomevent) >= 10000:
                pwm = 800
            else:
                pwm = 500
            if self.__x < randomevent:
                self.__p = pwm
            else:
                self.__p = -pwm
            self.__output(0, self.__p)
        for _ in range(3):
            self.__output(0, 0)

        # reset parameters
        print("Reset the real env!")
        # print("Press the bottom to reset the real env!")
        # while self.__k != 1:
        #     self.__input()

        self.sta_time = time.perf_counter()
        for _ in range(10):
            self.state, _, _, _ = self.step()
        self.t = 0

        self.steps_beyond_done = None

        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600
        carty = screen_height / 2  # TOP OF CART

        world_width = 10.65  # max visible position of cart
        scale = screen_width / world_width

        polewidth = scale * 0.05
        polelen = scale * 4.93  # 0.6 or self.l
        cartwidth = scale * 0.65
        cartheight = cartwidth / 3

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2

            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            cart.set_color(1, 0, 0)
            self.viewer.add_geom(cart)

            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0, 0, 1)
            self.poletrans = rendering.Transform(translation=(0, 0))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)

            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.1, 1, 1)
            self.viewer.add_geom(self.axle)

            # Make another circle on the top of the pole
            self.pole_bob = rendering.make_circle(polewidth / 2)
            self.pole_bob_trans = rendering.Transform()
            self.pole_bob.add_attr(self.pole_bob_trans)
            self.pole_bob.add_attr(self.poletrans)
            self.pole_bob.add_attr(self.carttrans)
            self.pole_bob.set_color(0, 0, 0)
            self.viewer.add_geom(self.pole_bob)

            self.wheel_l = rendering.make_circle(cartheight / 4)
            self.wheel_r = rendering.make_circle(cartheight / 4)
            self.wheeltrans_l = rendering.Transform(translation=(-cartwidth / 2, -cartheight / 2))
            self.wheeltrans_r = rendering.Transform(translation=(cartwidth / 2, -cartheight / 2))
            self.wheel_l.add_attr(self.wheeltrans_l)
            self.wheel_l.add_attr(self.carttrans)
            self.wheel_r.add_attr(self.wheeltrans_r)
            self.wheel_r.add_attr(self.carttrans)
            self.wheel_l.set_color(0, 0, 0)  # Black, (B, G, R)
            self.wheel_r.set_color(0, 0, 0)  # Black, (B, G, R)
            self.viewer.add_geom(self.wheel_l)
            self.viewer.add_geom(self.wheel_r)

            self.track = rendering.Line((0, carty - cartheight / 2 - cartheight / 4),
                                        (screen_width, carty - cartheight / 2 - cartheight / 4))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])
        self.pole_bob_trans.set_translation(-4.93 * np.sin(x[2]), 4.93 * np.cos(x[2]))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        self.__output(0, 0)


def control_model_init():
    pygame.init()
    canvas = pygame.display.set_mode((100, 100))
    canvas.fill((255, 255, 255))
    pygame.display.set_caption("control window")


if __name__ == '__main__':
    print("Program Start!")
    env = Env("cu.usbserial-142120")
    x = []
    y = []
    index = 0
    try:
        env.reset()
        while True:
            while True:
                # env.render()  # consume a lot of time
                observation, reward, done, _ = env.step()
                print(observation, reward, done)
                if done:
                    break
            env.reset()
    except:
        env.close()
        print("Program Exit!")

# if __name__ == '__main__':
#     control_model_init()
#     print("Program Start!")
#     env = Env("cu.usbserial-142120")
#     try:
#         env.reset(touch=True)
#         action = None  # random.randint(0, 2)
#         while True:
#             x = []
#             y = []
#             index = 0
#             while True:
#                 # env.render()  # consume a lot of time
#                 observation, reward, done, _ = env.step(action)
#                 print(observation)
#                 x.append(index)  # 添加i到x轴的数据中
#                 y.append(observation[2])  # 添加i的平方到y轴的数据中
#                 plt.clf()  # 清除之前画的图
#                 plt.plot(x, y)  # 画出当前x列表和y列表中的值的图形
#                 plt.pause(0.001)  # 暂停一段时间，不然画的太快会卡住显示不出来
#                 plt.ioff()  # 关闭画图窗口s
#                 index += 1
#                 for event in pygame.event.get():
#                     if event.type == pygame.KEYDOWN:
#                         if event.key == pygame.K_LEFT:
#                             action = 0
#                         elif event.key == pygame.K_RIGHT:
#                             action = 2
#                         elif event.key == pygame.K_SPACE:
#                             action = None
#
#                 if done:
#                     break
#             env.reset()
#             action = None  # random.randint(0, 2)
#     except:
#         env.close()
#         print("Program Exit!")
