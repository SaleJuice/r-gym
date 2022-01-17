# _*_ coding: utf-8 _*_
# @File        : cartpole_balance.py
# @Time        : 2021/10/17 19:09
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

from rgym.utils.easy_serial import WindowsBackground

np.set_printoptions(suppress=True)  # cancel scientific notation output


class RealCartPoleEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, portx):
        # hardware part
        self.ser = WindowsBackground()  # run this code in Windows11 OS
        assert (self.ser.connect(portx)), f"\n\tCan't connect real env by '{portx}'!\n\t\tThere only have ports: {self.ser.port_list}"

        # software part
        self.force_mag = 50
        self.discrete = True
        self.frequency = self.metadata['video.frames_per_second']  # the unit of frequency is "Hz"

        # condition to fail the episode
        self.theta_threshold_radians = 20 * (2 * math.pi / 360)
        self.x_threshold = 132 / 2  # the unit of x is "cm"

        high = np.array([
            self.x_threshold,
            np.finfo(np.float32).max,
            self.theta_threshold_radians,
            np.finfo(np.float32).max])

        if self.discrete:
            self.action_space = spaces.Discrete(5)
        else:
            self.action_space = spaces.Box(-self.force_mag, self.force_mag, shape=(1,))
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

        # time param
        self.sta_time = 0
        self.end_time = 0

        # offset
        self.__ox = 0
        self.__oa = 2048

        # private variables
        self.__px = 0  # previous position of cart
        self.__x = 0  # current position of cart
        self.__dx = 0  # delta position of cart

        self.__pa = 0  # previous angle of pole
        self.__a = 0  # current angle of pole
        self.__da = 0  # delta angle of pole

        self.__k = 0  # current value of push-button
        self.__p = 0  # output wait to send

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __input(self):
        num, res, _ = self.ser.read("all")
        if num >= 16 and res[res.rfind(')') - 14] == '(':
            res = res[res.rfind(')') - 13:res.rfind(')')]
            raw_data = list(map(int, res.split(',')))

            self.__k = raw_data[2]

            self.__x = (raw_data[0] - self.__ox) * self.x_threshold * 2 / 134200
            self.__dx = self.__x - self.__px
            self.__px = self.__x

            self.__a = (raw_data[1] - self.__oa) * math.pi * 2 / 4096
            self.__da = self.__a - self.__pa
            self.__pa = self.__a

            self.state = [self.__x, self.__dx, self.__a, self.__da]
            return True
        return False

    def __output(self, t, p):
        if self.ser.write(f"({t},{p},4)\n"):  # the frequency of real machine is 200Hz, 4 = 200 / 50
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
                    self.__p = -1 * self.force_mag
                elif action == 1:
                    self.__p = -0.5 * self.force_mag
                elif action == 2:
                    self.__p = 0
                elif action == 3:
                    self.__p = 0.5 * self.force_mag
                elif action == 4:
                    self.__p = 1 * self.force_mag
            else:
                self.__p = action

            self.__p = max(-self.force_mag, min(self.__p, self.force_mag))
            # self.__p = max(-1800, min(self.__p, 1800))
            self.__output(1, self.__p)

        # control frequency of env
        if freq is None:
            freq = self.frequency
        # print(time.perf_counter() - self.sta_time)
        while (self.end_time - self.start_time) < (1 / freq):
            self.end_time = time.perf_counter()
        # print(time.perf_counter() - self.sta_time)
        self.start_time = time.perf_counter()

        self.__input()

        done = bool(
            self.state[0] < -self.x_threshold * 0.9
            or self.state[0] > self.x_threshold * 0.9
            or self.state[2] < -self.theta_threshold_radians
            or self.state[2] > self.theta_threshold_radians
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
            self.__output(0, 0)

        print(np.array(self.state), reward, done, {})
        return np.array(self.state), reward, done, {}

    def reset(self, touch=False):
        self.__turns = 0
        self.__pa = math.pi  # previous angle of pole
        self.__a = 0  # current angle of pole
        self.__da = 0  # delta angle of pole
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

        # print("Reset the real env!")
        print("Press the bottom to reset the real env!")
        while self.__k != 2:
            self.__input()

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


if __name__ == '__main__':
    pass



