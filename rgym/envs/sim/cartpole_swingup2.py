# _*_ coding: utf-8 _*_
# @File        : cartpole_swingup2.py
# @Time        : 2021/12/6 16:45
# @Author      : SaleJuice
# @E-Mail      : linxzh@shanghaitech.edu.cn
# @Institution : LIMA Lab, ShanghaiTech University, China
# @SoftWare    : PyCharm


import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

np.set_printoptions(suppress=True)  # cancel scientific notation output


class Env(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self):
        # software part
        self.discrete = True
        self.frequency = self.metadata['video.frames_per_second']  # the unit of frequency is "Hz"

        # simulate part
        self.g = 9.82  # gravity
        self.m_c = 0.5  # cart mass
        self.m_p = 0.5  # pendulum mass
        self.total_m = (self.m_p + self.m_c)
        self.l = 0.6  # pole's length
        self.m_p_l = (self.m_p * self.l)
        self.force_mag = 10.0
        self.dt = 1 / self.frequency  # seconds between state updates
        self.b = 3  # friction coefficient

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        high = np.array([
            self.x_threshold,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])

        if self.discrete:
            self.action_space = spaces.Discrete(3)
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
        self.t_limit = 1000

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action=None, freq=None):
        if action is None:
            action = 0
        else:
            if self.discrete:
                # valid action
                err_msg = "%r (%s) invalid" % (action, type(action))
                assert self.action_space.contains(action), err_msg
                if action == 0:
                    action = -self.force_mag
                elif action == 1:
                    action = 0
                else:
                    action = self.force_mag
            else:
                action = np.clip(action, -1.0, 1.0)[0]
                action *= self.force_mag

        state = self.state
        x, x_dot, theta, theta_dot = state

        s = math.sin(theta)
        c = math.cos(theta)

        xdot_update = (-2 * self.m_p_l * (theta_dot ** 2) * s + 3 * self.m_p * self.g * s * c + 4 * action - 4 * self.b * x_dot) / (4 * self.total_m - 3 * self.m_p * c ** 2)
        thetadot_update = (-3 * self.m_p_l * (theta_dot ** 2) * s * c + 6 * self.total_m * self.g * s + 6 * (action - self.b * x_dot) * c) / (4 * self.l * self.total_m - 3 * self.m_p_l * c ** 2)
        x = x + x_dot * self.dt
        theta = theta + theta_dot * self.dt
        x_dot = x_dot + xdot_update * self.dt
        theta_dot = theta_dot + thetadot_update * self.dt

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            self.state[0] < -self.x_threshold * 0.9
            or self.state[0] > self.x_threshold * 0.9
            or self.t >= self.t_limit
        )

        self.t += 1

        # compute costs - saturation cost
        goal = np.array([0.0, self.l])
        pole_x = self.l * np.sin(theta)
        pole_y = self.l * np.cos(theta)
        position = np.array([self.state[0] + pole_x, pole_y])
        squared_distance = np.sum((position - goal) ** 2)
        squared_sigma = 0.6 ** 2
        costs = 1 - np.exp(-0.5 * squared_distance / squared_sigma)

        # reward methods
        reward = math.cos(self.state[2])

        # print(np.array(self.state), -costs, done, action)
        return np.array(self.state), -costs, done, {}

    def reset(self):
        # self.state = self.np_random.normal(loc=np.array([0.0, 0.0, 30*(2*np.pi)/360, 0.0]), scale=np.array([0.0, 0.0, 0.0, 0.0]))
        self.state = self.np_random.normal(loc=np.array([0.0, 0.0, np.pi, 0.0]), scale=np.array([0.02, 0.02, 0.02, 0.02]))
        self.steps_beyond_done = None
        self.t = 0  # timestep
        return np.array(self.state)

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 600  # before was 400

        world_width = 5  # max visible position of cart
        scale = screen_width / world_width
        carty = screen_height / 2  # TOP OF CART
        polewidth = 6.0
        polelen = scale * self.l  # 0.6 or self.l
        cartwidth = 40.0
        cartheight = 20.0

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
        self.poletrans.set_rotation(x[2])
        self.pole_bob_trans.set_translation(-self.l * np.sin(x[2]), self.l * np.cos(x[2]))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        pass


if __name__ == '__main__':
    print("Program Start!")
    env = Env()
    try:
        env.reset()
        action = None
        while True:
            env.render()
            observation, reward, done, _ = env.step(action)
            print(observation)
    except:
        env.close()
        print("Program Exit!")