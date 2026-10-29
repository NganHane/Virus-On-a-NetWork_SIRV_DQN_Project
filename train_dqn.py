import os
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from env import MesaEpiEnv


def make_env(params):
    return lambda: MesaEpiEnv(model_params=params)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--timesteps', type=int, default=200000)
    p.add_argument('--save_path', type=str, default='saved_dqn')
    p.add_argument('--N', type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    model_params = {
        'N': args.N,
        'initial_infected': 10,
        'p_transmission': 0.08,
        'infectious_period': 14,
        'daily_capacity_fraction': 0.02,
        'post_vacc_period_days': 12,
        'post_vacc_infection_multiplier': 0.2,
    }

    env = DummyVecEnv([make_env(model_params)])

    os.makedirs(args.save_path, exist_ok=True)

    model = DQN('MlpPolicy', env,
                learning_rate=1e-4,
                buffer_size=50000,
                learning_starts=1000,
                batch_size=64,
                gamma=0.99,
                train_freq=4,
                target_update_interval=1000,
                verbose=1,
                tensorboard_log=args.save_path)

    model.learn(total_timesteps=args.timesteps)
    model.save(os.path.join(args.save_path, 'dqn_sirv_model'))
    print('Saved model to', args.save_path)

if __name__ == '__main__':
    main()