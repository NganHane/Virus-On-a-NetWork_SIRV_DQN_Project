import matplotlib.pyplot as plt
import numpy as np
from env import MesaEpiEnv
from stable_baselines3 import DQN
from model import EpidemicModel

def run_policy(env, model, max_steps=365):
    obs = env.reset()
    history = {'I':[], 'V':[], 'reward':[], 'action':[]}
    for t in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        I = sum(1 for a in env.model.agents.values() if a.status == 'I')
        V = sum(1 for a in env.model.agents.values() if a.protected or a.status == 'V')
        history['I'].append(I)
        history['V'].append(V)
        history['reward'].append(reward)
        history['action'].append(int(action))
        if done:
            break
    return history

def plot_history(history, title='Policy run'):
    fig, ax = plt.subplots(3,1, figsize=(8,8), sharex=True)
    ax[0].plot(history['I'], label='I (active)')
    ax[0].legend()
    ax[1].plot(history['V'], label='V (protected)')
    ax[1].legend()
    ax[2].plot(history['reward'], label='reward')
    ax[2].legend()
    ax[2].set_xlabel('days')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    model_params = {
        'N': 1000,
        'initial_infected': 10,
        'p_transmission': 0.04,
        'infectious_period': 14,
        'daily_capacity_fraction': 0.02,
        'post_vacc_period_days': 12,
        'post_vacc_infection_multiplier': 0.2,
    }

    max_days = 35
    
    model = DQN.load('saved_dqn/dqn_sirv_model')
    env = MesaEpiEnv(model_params=model_params)
    history = run_policy(env, model)
    plot_history(history, title='RL policy')



