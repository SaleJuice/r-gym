# _*_ coding: utf-8 _*_
# @File        : real_cartpole_swingup.py
# @Time        : 2021/12/7 10:29
# @Author      : SaleJuice
# @E-Mail      : linxzh@shanghaitech.edu.cn
# @Institution : LIMA Lab, ShanghaiTech University, China
# @SoftWare    : PyCharm


import time
import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

import random
from easyserial import WindowsBackground
import matplotlib.pyplot as plt
import pygame

np.set_printoptions(suppress=True)  # cancel scientific notation output


class RealCartPoleSwingUpEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, portx):
        # hardware part
        self.ser = WindowsBackground()  # run this code in Windows11 OS
        assert (self.ser.connect(portx)), f"\n\tCan't connect real env by '{portx}'!\n\t\tThere only have ports: {self.ser.port_list}"

        # software part
        self.force_mag = 150
        self.discrete = True
        self.frequency = self.metadata['video.frames_per_second']  # the unit of frequency is "Hz"

        # condition to fail the episode
        # self.theta_threshold_radians = 180 * (2 * math.pi / 360)
        self.x_threshold = 132 / 2  # the unit of x is "cm"

        high = np.array([
                self.x_threshold,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max])

        if self.discrete:
            self.action_space = spaces.Discrete(2)
        else:
            self.action_space = spaces.Box(-self.force_mag, self.force_mag, shape=(1,))
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

        # time param
        self.start_time = 0
        self.end_time = 0

        self.t = 0  # timestep
        self.t_limit = 250

        # offset
        self.__ox = 0
        self.__oa = 2048
        self.__turns = 0  # number of turns

        # private variables
        self.__px = 0  # previous position of cart
        self.__x = 0  # current position of cart
        self.__dx = 0  # delta position of cart

        self.__pa = math.pi  # previous angle of pole
        self.__a = 0  # current angle of pole
        self.__da = 0  # delta angle of pole

        self.__k = 0  # current value of push-button
        self.__p = 0  # output wait to send

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __input(self):
        num, res, _ = self.ser.read("all")
        if num >= 16 and res[res.rfind(')')-14] == '(':
            res = res[res.rfind(')')-13:res.rfind(')')]
            raw_data = list(map(int, res.split(',')))

            self.__k = raw_data[2]

            self.__x = (raw_data[0] - self.__ox) * self.x_threshold * 2 / 134200
            self.__dx = self.__x - self.__px
            self.__px = self.__x

            self.__a = (raw_data[1] - self.__oa) * math.pi * 2 / 4096 + self.__turns * math.pi * 2
            if self.__pa - self.__a > math.pi:
                self.__a += math.pi * 2
                self.__turns += 1
            elif self.__pa - self.__a < -math.pi:
                self.__a -= math.pi * 2
                self.__turns -= 1
            self.__da = self.__a - self.__pa
            self.__pa = self.__a

            self.state = [self.__x, self.__dx, self.__a, self.__da]
            return True
        return False

    def __output(self, t, p):
        t = int(t)
        p = int(p)
        f = int(200 / self.frequency)
        if self.ser.write(f"({t},{p},{f})\n"):  # the frequency of real machine is 200Hz, 4 = 200 / 50
            return True
        return False

    def step(self, action=None, freq=None):
        if action is None:
            self.__output(0, 0)
        else:
            # valid action
            err_msg = "%r (%s) invalid" % (action, type(action))
            assert self.action_space.contains(action), err_msg

            if self.discrete:
                if action == 0:
                    self.__p -= self.force_mag
                elif action == 1:
                    self.__p += self.force_mag
            else:
                self.__p = action

            # self.__p = max(-self.force_mag, min(self.__p, self.force_mag))
            self.__p = max(-1800, min(self.__p, 1800))
            self.__output(0, self.__p)

        # control frequency of env
        if freq is None:
            freq = self.frequency
        # print(time.perf_counter() - self.start_time)
        while (self.end_time - self.start_time) < (1 / freq):
            self.end_time = time.perf_counter()
        # print(time.perf_counter() - self.start_time)
        self.start_time = time.perf_counter()

        self.__input()

        done = bool(
            self.state[0] < -self.x_threshold * 0.7
            or self.state[0] > self.x_threshold * 0.7
            # or self.t >= self.t_limit
        )

        self.t += 1

        if done:
            self.__output(0, 0)
            # reward methods
            reward = -1000
        else:
            # reward methods
            reward = math.cos(self.state[2])

        # compute costs - saturation cost
        goal = np.array([0.0, 30])
        pole_x = 30 * np.sin(self.state[2])
        pole_y = 30 * np.cos(self.state[2])
        position = np.array([self.state[0] + pole_x, pole_y])
        squared_distance = np.sum((position - goal) ** 2)
        squared_sigma = 20 ** 2
        costs = 1 - np.exp(-0.5 * squared_distance / squared_sigma)

        print(np.array(self.state), -costs, done, {})
        return np.array(self.state), -costs, done, {}

    def reset(self, touch=False):
        self.__turns = 0
        self.__pa = math.pi  # previous angle of pole
        self.__a = 0  # current angle of pole
        self.__da = 0  # delta angle of pole
        self.t = 0  # timestep
        if touch:
            # touch edge
            self.__input()
            while self.__k != 1:
                self.__input()
                self.__output(0, -800)
            self.__ox = (self.__x * 134200 / (self.x_threshold * 2)) + 67100  # 66569
            self.__output(0, 800)

        # back to the middle
        self.__input()
        randomevent = self.np_random.uniform(low=-self.x_threshold * 0.1, high=self.x_threshold * 0.1)
        while abs(self.__x - randomevent) >= 0.01:
            self.__input()
            if abs(self.__x - randomevent) >= self.x_threshold * 2 * 0.1:
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

        print("Reset the real env!")
        # print("Press the bottom to reset the real env!")
        # while self.__k != 1:
        #     self.__input()

        # return
        self.__input()
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
        self.__output(0, 0)


def control_model_init():
    pygame.init()
    canvas = pygame.display.set_mode((100, 100))
    canvas.fill((255, 255, 255))
    pygame.display.set_caption("control window")


if __name__ == '__main__':
    control_model_init()
    print("Program Start!")
    env = RealCartPoleSwingUpEnv("COM5")
    try:
        env.reset(touch=True)
        action = None  # random.randint(0, 2)
        while True:
            x = []
            y = []
            index = 0
            while True:
                # env.render()  # consume a lot of time
                observation, reward, done, _ = env.step(action)
                print(observation)
                x.append(index)  # 添加i到x轴的数据中
                y.append(observation[2])  # 添加i的平方到y轴的数据中
                plt.clf()  # 清除之前画的图
                plt.plot(x, y)  # 画出当前x列表和y列表中的值的图形
                plt.pause(0.001)  # 暂停一段时间，不然画的太快会卡住显示不出来
                plt.ioff()  # 关闭画图窗口s
                index += 1
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            action = 0
                        elif event.key == pygame.K_RIGHT:
                            action = 2
                        elif event.key == pygame.K_SPACE:
                            action = None
                if done:
                    break
            env.reset()
            action = None  # random.randint(0, 2)
    except:
        env.close()
        print("Program Exit!")
