from   real.pendulum import *

import matplotlib.pyplot as plt
import os
import time
import numpy as np

class RealPendulum:
    def __init__(self, target_q=0.0, n_seconds=1, action_frequency=200):
        # Initialize all that needed for real robot
        self._param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO))
        os.sched_setscheduler(0, os.SCHED_FIFO, self._param)
        self._bus = can.interface.Bus(bustype='socketcan', channel='can0', bitrate=1000000)
        self._motor = Actuator(0x141, self._bus)
        self._pendulum = Pendulum()

        # Define episode parameters
        self.n_seconds = n_seconds
        self.action_frequency = action_frequency
        self.target_q = target_q

    def reset(self):
        # Move the controller to the target position
        state      = self._motor.get_state()
        q, dq, ddq = state.X[0][0], state.X[1][0], 0.0
        prev_time  = time.time()
        q_des = 0.0
        u     = 0.0
        n     = 0
        while np.abs(q) > 1e-2 or np.abs(dq) > 1e-2:
            # Wait until we should act
            while time.time() - prev_time < 0.01:
                pass
            prev_time = time.time()

            # Find control signal using pd
            X_des = np.array([[q_des],
                            [0],
                            [0]])
            X = state.X
            u = self._pendulum.control_pd(X_des, X, 1.0, 0.1)

            # Send the signal to the motor
            state      = self._motor.send_current(u)
            ddq       += (np.abs((state.X[1][0] - dq) / (0.01)) - ddq) / (n+1)
            q, dq      = state.X[0][0], state.X[1][0]
            n         += 1

        # Reset the current
        state = self._motor.send_current(0)
        prev_time = time.time()
        while time.time() - prev_time < 0.2:
            pass

        # Start acting
        self._prev_time = time.time()
        self._start_time = self._prev_time
        self._observation = None

        return self._build_observation(state)

    def step(self, action):
        # Wait until we should act
        while time.time() - self._prev_time < 1.0 / self.action_frequency:
            pass
        self._prev_time = time.time()

        # Act and wait for the response
        raw_state = self._motor.send_current(action)

        # Construct an observation
        obs = self._build_observation(raw_state)

        # End the simulation
        done = False
        if time.time() - self._start_time >= self.n_seconds:
            self._motor.send_current(0)
            done = True

        return obs, done

    def cost(self, observations, cost="total-energy"):
        if cost != "total-energy":
            raise NotImplementedError()
        
        # Target energy (should be matched at every timestep)
        target_energy = np.sin(self.target_q) * self._pendulum.l * self._pendulum.m * self._pendulum.g

        # Compute quadratic cost
        total_cost = 0.0
        for observation in observations:
            q, dq = observation[0], observation[1]
            energy = 0.5 * self._pendulum.I * dq**2 + self._pendulum.m * self._pendulum.g * np.sin(q) * self._pendulum.l
            total_cost += (target_energy - energy) ** 2

        return total_cost

    def _build_observation(self, raw_state):
        if self._observation is None:
            self._observation = [raw_state.X[0][0] + 1.5*np.pi, raw_state.X[1][0]] * 4
        else:
            for i in range(3, 0, -1):
                self._observation[i*2] = self._observation[(i-1)*2]
                self._observation[i*2+1] = self._observation[(i-1)*2+1]
            self._observation[0] = raw_state.X[0][0] + 1.5*np.pi
            self._observation[1] = raw_state.X[1][0]
        
        return self._observation

    def close(self):
        print("am i dead?")
        self._motor.send_current(0)

class RealPolicy:
    INPUT_SHAPE = 8
    INTERNAL_SHAPE = 64
    OUTPUT_SHAPE = 1

    def __init__(self, tanh=False):
        self.w1 = np.zeros(shape=(self.INTERNAL_SHAPE, self.INPUT_SHAPE))
        self.b1 = np.zeros(shape=(self.INTERNAL_SHAPE))
        self.w2 = np.zeros(shape=(self.INTERNAL_SHAPE, self.INTERNAL_SHAPE))
        self.b2 = np.zeros(shape=(self.INTERNAL_SHAPE))
        self.w3 = np.zeros(shape=(self.OUTPUT_SHAPE, self.INTERNAL_SHAPE))
        self.b3 = np.zeros(shape=(self.OUTPUT_SHAPE))
        self._tanh = tanh

    def forward(self, obs):
        h = np.tanh(self.w1 @ obs + self.b1)
        h = np.tanh(self.w2 @ h + self.b2)
        h = self.w3 @ h + self.b3

        if self._tanh:
            return np.tanh(h)
        else:
            return np.clip(h, -1, 1)

    def set_weights(self, weights):
        shapes = [self.w1.shape, self.w2.shape, self.w3.shape, self.b1.shape, self.b2.shape, self.b3.shape]
        result = []
        prev_ind = 0
        for i in range(len(shapes)):
            result.append(weights[prev_ind:prev_ind+np.prod(shapes[i])].reshape(shapes[i]))
            prev_ind += np.prod(shapes[i])

        self.w1 = result[0]
        self.w2 = result[1]
        self.w3 = result[2]
        self.b1 = result[3]
        self.b2 = result[4]
        self.b3 = result[5]


# Simple test
if __name__ == "__main__":
    env = RealPendulum()
    for e in range(10):
        obs          = env.reset()
        print("really?")
        observations = [obs]
        done         = False
        while not done:
            obs, done = env.step(np.random.randn() * 5)
            observations.append(obs)
        print(f"Episode: {e}; Time: {time.asctime()}; Cost: {env.cost(observations)}")