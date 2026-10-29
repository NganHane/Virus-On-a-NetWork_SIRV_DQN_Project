import gym
import numpy as np
from gym import spaces
from model import EpidemicModel

class MesaEpiEnv(gym.Env):
    def __init__(self, model_params=None):
        super().__init__()
        if model_params is None:
            model_params = {}
        self.model_params = model_params
        self.model = None
        self.action_space = spaces.Discrete(3)  # 0=open,1=partial,2=full
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

    def reset(self):
        self.model = EpidemicModel(**self.model_params)
        obs = self.model.get_observation()
        return obs

    def step(self, action):
        # apply action (lockdown level) and step one day
        self.model.step(action)
        obs = self.model.get_observation()
        reward = float(self.model.reward)
        done = False
        if self.model.step_count >= self.model_params.get('max_days', 365):
            done = True
        I = sum(1 for a in self.model.agents.values() if a.status == 'I')
        if I == 0:
            done = True
        info = {
            'cum_infected': self.model.cum_infected,
            'cum_vaccinated': self.model.cum_vaccinated,
            'day': self.model.step_count,
        }
        return obs, reward, done, info

    def render(self, mode='human'):
        I = sum(1 for a in self.model.agents.values() if a.status == 'I')
        S = sum(1 for a in self.model.agents.values() if a.status == 'S')
        R = sum(1 for a in self.model.agents.values() if a.status == 'R')
        V = sum(1 for a in self.model.agents.values() if a.protected or a.status == 'V')
        print(f"Day {self.model.step_count}: S={S} I={I} R={R} V={V} Restr={self.model.restriction} Reward={self.model.reward:.4f}")