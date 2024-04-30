'''
This is the file to compare the reward of the standard model and the extracted model.

@Author: Xiangyun Rao
@Date: 2023.7.30
'''
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from agent import DQN, ExtractNet
from tool import plot_reward

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
    # state, _ = env1.reset(seed=SEED)
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

def compare_different_models(standard_agent, compare_agent):
    standard_agent.eval()
    compare_agent.eval()
    standard_rewards = []
    compare_rewards = []
    evaluation_times = 100
    # evaluate 100 times and get the average reward
    for _ in tqdm(range(evaluation_times)):
        standard_rewards.append(evaluation(standard_agent))
        compare_rewards.append(evaluation(compare_agent))
    # get model name
    standard_name = standard_agent.__class__.__name__
    compare_name = compare_agent.__class__.__name__
    # get the y-axis limit
    y_lim = min(min(standard_rewards), min(compare_rewards)) - 50
    # plot
    plot_reward(standard_rewards, standard_name + "_reward_{}_evaluations".format(evaluation_times), y_lim)
    plot_reward(compare_rewards, compare_name + "_reward_{}_evaluations".format(evaluation_times), y_lim)
    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    standard_agent = DQN(input_n, discrete_action_n).to(device)
    compare_agent = ExtractNet(input_n, discrete_action_n).to(device)

    standard_agent.load_state_dict(torch.load('./weights/target_model_{}_{}.pt'.format(ENV_NAME, SEED)))
    compare_agent.load_state_dict(torch.load('./weights/ExtractNet_{}_{}.pt'.format(ENV_NAME, SEED)))

    compare_different_models(standard_agent, compare_agent)
