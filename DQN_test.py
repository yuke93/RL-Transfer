import os
import os.path as osp
import gym
import glob
import numpy as np
from timeit import default_timer as timer
import torch

from utils.wrappers import *
from utils.hyperparameters import Config

from model import Model

config = Config()
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.MAX_FRAMES = 500000  # how long is the model trained
config.VALID_EPISODES = 50  # how many episodes to test

# test settings
student_path = 'pretrained_weights/'
test_dir = 'log/test/'

if __name__ == '__main__':

    os.makedirs(test_dir, exist_ok=True)

    start = timer()

    valid_env_name_list = ["AirRaidNoFrameskip-v4", "CarnivalNoFrameskip-v4",
                           "DemonAttackNoFrameskip-v4", "AssaultNoFrameskip-v4"]

    env_names = [0, 1, 2, 3]
    env_name_list = [valid_env_name_list[i] for i in env_names]

    # make student env
    student_env_list = list()
    for i, env_name in enumerate(env_name_list):
        env = make_atari(env_name)
        student_dir = 'log'
        # env = bench.Monitor(env, os.path.join(student_dir, env_name))
        env = wrap_deepmind(env, episode_life=True,
                            clip_rewards=False, frame_stack=False, scale=True)
        env = WrapPyTorch(env)
        student_env_list.append((env_name, env))

    gpu_id = 0
    with torch.cuda.device(gpu_id):


        for env_id, (env_name, test_env) in enumerate(student_env_list):

            # load model
            model = Model(env=test_env, config=config)
            model_weights = osp.join(student_path, env_name+'.dump')
            model.model.load_state_dict(torch.load(model_weights))

            # test
            print('[Test] env_id: {}, No. frame: {}'.format(env_id, config.MAX_FRAMES))
            num_done = 0
            test_observation = test_env.reset()
            test_reward_list = []
            steps_list = []
            test_episode_reward = 0.
            steps = 0
            while num_done < config.VALID_EPISODES:
                test_action = model.get_action(test_observation, env_id)
                test_observation, test_reward, done, _ = test_env.step(test_action)
                test_episode_reward += test_reward
                steps += 1
                # print(reward, done)
                if done:
                    test_observation = test_env.reset()
                    num_done += 1
                    test_reward_list.append(test_episode_reward)
                    steps_list.append(steps)
                    print('episode {}: steps {}, total reward {}'.format(num_done, steps, test_episode_reward))
                    test_episode_reward = 0.
                    steps = 0
            test_mean_reward = np.asarray(test_reward_list).mean()
            mean_steps = np.asarray(steps_list).mean()
            print('[RESULTS] env: {}, mean steps {}, mean reward {}'.format(env_name, mean_steps, test_mean_reward))
            #model.save_val_res(test_dir, env_name, mean_steps, test_mean_reward)

