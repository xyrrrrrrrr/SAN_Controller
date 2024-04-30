"""
This is the file to compare the control performance of the systems.

@Author: Xiangyun Rao
@Date: 2024.4.16
"""

import math
import random
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm, trange

from buffer import ReplayBuffer
from agent import ExtractNet, SANExtractor
from tool import update_target, compare_result, plot_reward


def set_env(env_id):
    global env, env1, discrete_action_n, input_n, faction
    env = gym.make(env_id)
    env1 = gym.make(env_id)
    input_n = env.observation_space.shape[0]
    if not isinstance(env.action_space, gym.spaces.discrete.Discrete):
        discrete_action_n = 11  # [0, discrete_action_n-1]
        action_lowbound = env.action_space.low[0]
        action_upbound = env.action_space.high[0]

        #discrete action to continuous action
        def faction(discrete_n):
            return action_lowbound + (discrete_n / (discrete_action_n - 1)) * (action_upbound - action_lowbound)
        
    else:
        discrete_action_n = env.action_space.n


def evaluation(current_model):
    state = env1.reset()
    env1.seed(SEED)
    done = False
    r = 0
    while not done:
        action = current_model.act(state, 0, device, discrete_action_n)
        if not isinstance(env.action_space, gym.spaces.discrete.Discrete):
            next_state, reward, done, _ = env1.step(np.array([faction(action)]))
        else:
            next_state, reward, done, _ = env1.step(action)
        state = next_state
        r += reward

    return r

if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    SEED = 9
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    gamma = 0.98

    env, env1 = None, None
    discrete_action_n, input_n = 0, 0
    faction = None

    ENV_NAME = "Pendulum-v0"
    set_env(ENV_NAME)

    num_frames = 50000
    # num_frames = 200000
    epsilon_start = 0.5
    epsilon_final = 0.01
    epsilon_decay = 1500
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
    update_target_step = 100
    learning_rate = 0.01
    eval_step = 200
    h_size=24



    # set weights

    weights = dict()
    weights['extractor'] = "./weights/ExtractNet_Pendulum-v0_9.pt"
    weights['SANController'] = "./weights/SANController_Pendulum-v0_9.pt"

    # set model
    # origin_model = ExtractNet(input_n, discrete_action_n, h_size=h_size).to(device)
    # san_model = SANExtractor(input_n, discrete_action_n, h_size=h_size).to(device)

    model1= ExtractNet(input_n, discrete_action_n, h_size=h_size).to(device)
    model2 = ExtractNet(input_n, discrete_action_n, h_size=h_size).to(device)

    # select models from lists

    origin_model = model1
    san_model = model2
    origin_model.load_state_dict(torch.load(weights['extractor']))
    san_model.load_state_dict(torch.load(weights['extractor']))

    # training the models

    for mod    # load weights
    origin_model.load_state_dict(torch.load(weights['extractor']))
    san_model.load_weights(weights)

    # evaluation
    rewards = []
    for mod in [origin_model, san_model]:
        mod.eval()
        reward = []
        for _ in trange(1000):
            reward.append(evaluation(mod))
        rewards.append(reward)
        plot_reward(reward, mod.__class__.__name__ + "_reward", y_lim=-50)

    # compare the performance distance
    diff = np.array(rewards[0]) - np.array(rewards[1])
    diff = [i for i in diff if i < 100]
    diff = sum(diff) / len(diff)
    print("The difference between the two models is: ", diff)

