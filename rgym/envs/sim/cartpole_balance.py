# _*_ coding: utf-8 _*_
# @File        : cartpole_balance.py
# @Time        : 2021/12/7 10:16
# @Author      : SaleJuice
# @E-Mail      : linxzh@shanghaitech.edu.cn
# @Institution : LIMA Lab, ShanghaiTech University, China
# @SoftWare    : PyCharm

import math
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding


class Env(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, t_limit=float('inf')):
        # kinematic parameters
        self.g = 9.8  # N/kg
        self.m_c = 1.0  # kg
        self.m_p = 0.1  # kg
        self.l = 1.0 / 2  # m

        self.fri_c = 1.0
        self.fri_p = 0.1

        self.dt = 0.02  # s

        self.f_max = 10.0  # N

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
        self.position_threshold = 2.4  # m

        self.angle_pre = 0
        self.angle_cur = 0
        self.angle_delta = 0
        self.angle_threshold = (12 / 360) * (2 * math.pi)  # rad

        low = np.array(
            [
                -self.position_threshold,
                -np.finfo(np.float32).max,
                -self.angle_threshold,
                -np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        high = np.array(
            [
                self.position_threshold,
                np.finfo(np.float32).max,
                self.angle_threshold,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # others
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def kinematic(self, f):
        m_all = self.m_c + self.m_p
        sin_theta = math.sin(self.theta)
        cos_theta = math.cos(self.theta)

        # acceleration
        if True:
            # reference with https://coneural.org/florian/papers/05_cart_pole.pdf
            theta_acc = (self.g * sin_theta + cos_theta * ((- f - self.m_p * self.l * self.theta_dot ** 2 * sin_theta) / m_all)) / (self.l * (4 / 3 - (self.m_p * cos_theta ** 2) / m_all))
            x_acc = (f + self.m_p * self.l * (self.theta_dot ** 2 * sin_theta - theta_acc * cos_theta)) / m_all
        elif False:
            # reference with https://ieeexplore.ieee.org/document/9659410
            theta_acc = (-3 * self.m_p * self.l * self.theta_dot ** 2 * sin_theta * cos_theta + 6 * m_all * self.g * sin_theta + 4 * f + 6 * (f) * cos_theta) / (self.l * (4 * m_all - 3 * self.m_p * cos_theta ** 2))
            x_acc = (-2 * self.m_p * self.l * self.theta_dot ** 2 * sin_theta + 3 * self.m_p * self.g * sin_theta * cos_theta + 4 * f) / (4 * m_all - 3 * self.m_p * cos_theta ** 2)
        else:
            # reference with https://zhuanlan.zhihu.com/p/358140662
            theta_acc = ((m_all * self.g - self.m_p * self.l * 2 * self.theta_dot ** 2 * cos_theta) * sin_theta + (f * cos_theta)) / (self.l * 2 * (self.m_c + self.m_p * sin_theta ** 2))
            x_acc = (self.m_p * sin_theta * (self.l * 2 * self.theta_dot ** 2 - self.g * cos_theta) + f) / (self.m_c + self.m_p * sin_theta ** 2)

        # friction
        theta_acc += -self.fri_p * self.theta_dot
        x_acc += -self.fri_c * self.x_dot

        # velocity
        self.x_dot += x_acc * self.dt
        self.theta_dot += theta_acc * self.dt

        # position
        self.x += self.x_dot * self.dt
        self.theta += self.theta_dot * self.dt

    def feedback(self):
        self.t += 1

        done = bool(
            self.position_cur < -self.position_threshold
            or self.position_cur > self.position_threshold
            or self.angle_cur < -self.angle_threshold
            or self.angle_cur > self.angle_threshold
            or self.t > self.t_limit
        )

        # distance
        goal = np.array([0.0, self.l * 2])
        pole_x = self.l * 2 * np.sin(self.angle_cur)
        pole_y = self.l * 2 * np.cos(self.angle_cur)
        position = np.array([self.position_cur + pole_x, pole_y])
        squared_distance = np.sum((position - goal) ** 2)  # max(squared_distance) = self.l * 4
        squared_sigma = 0.5 ** 2
        cost = np.exp(-0.5 * squared_distance / squared_sigma)
        # cost = 1 / (squared_distance + 1)

        # angle
        # goal = np.array([self.position_cur, self.l * 2])
        # pole_x = self.l * 2 * np.sin(self.angle_cur)
        # pole_y = self.l * 2 * np.cos(self.angle_cur)
        # position = np.array([self.position_cur + pole_x, pole_y])
        # squared_distance = np.sum((position - goal) ** 2)  # max(squared_distance) = self.l * 4
        # squared_sigma = 0.5 ** 2
        # cost = np.exp(-0.5 * squared_distance / squared_sigma)
        # # cost = 1 / (squared_distance + 1)

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

        return reward, done

    def step(self, action=None):
        if action != None:
            assert self.action_space.contains(action), f"action = {action} ({type(action)}) is invalid"

        if action == 0:
            force = -self.f_max
        elif action == 1:
            force = self.f_max
        else:
            force = 0

        self.kinematic(force)

        # TODO: gaussian noise part: np.random.normal(loc=0, scale=0.00157)
        self.position_cur = self.x
        self.angle_cur = self.theta % (2 * math.pi)
        if self.angle_cur > math.pi:
            self.angle_cur -= 2 * math.pi
        self.position_delta = self.position_cur - self.position_pre
        self.angle_delta = self.angle_cur - self.angle_pre
        if self.angle_delta > math.pi:
            self.angle_delta -= 2 * math.pi
        elif self.angle_delta < -math.pi:
            self.angle_delta += 2 * math.pi
        self.position_pre = self.position_cur
        self.angle_pre = self.angle_cur

        self.state = (self.position_cur, self.position_delta, self.angle_cur, self.angle_delta)

        reward, done = self.feedback()

        return np.array(self.state), round(reward, 8), done, {'r': reward, 'f': force}

    def reset(self):
        self.position_pre, self.angle_pre = 0, 0
        self.x, self.x_dot, self.theta, self.theta_dot = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))

        self.step()

        self.t = 0
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
    np.set_printoptions(precision=8, suppress=True, sign='+')  # cancel scientific notation output
    env = Env()
    while True:
        env.reset()
        while True:
            env.render()
            obs, rew, done, info = env.step()
            print(obs, rew, done, info)
            if done: break
