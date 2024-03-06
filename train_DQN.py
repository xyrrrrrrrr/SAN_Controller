'''
This is the file to train the DQN agent.

@Author: Xiangyun Rao
@Date: 2023.7.13
'''
import math
import random
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from buffer import ReplayBuffer
from agent import DQN
from tool import update_target, compare_result

def learn(batch_size, current_model, target_model, replay_buffer, optimizer, step=1):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    with torch.no_grad():
        state = torch.FloatTensor(np.float32(state)).to(device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(device)
        action = torch.LongTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(done).to(device)
    q_values = current_model(state)
    next_q_values = current_model(next_state)
    if USE_TARGET_NET:
        next_q_values_target = target_model(next_state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    
    if USE_DOUBLE_Q:
        # ddqn的核心就是用当前网络选择动作，用target网络计算q值
        action = torch.max(next_q_values, 1)[1].unsqueeze(1)
        next_q_value = next_q_values_target.gather(1, action).squeeze(1)
    else:
        if USE_TARGET_NET:
            # 直接根据target网络计算q值
            action = torch.max(next_q_values_target, 1)[1].unsqueeze(1)
            next_q_value = next_q_values_target.gather(1, action).squeeze(1)
        else: # the computation of the Q values of next states for DQN w/o using target networks
            action = torch.max(next_q_values, 1)[1].unsqueeze(1)
            next_q_value = next_q_values.gather(1, action).squeeze(1)

    # 根据bellman方程计算
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    loss = F.mse_loss(q_value, expected_q_value.detach()).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


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




def push_buffer(buffers, sample ,step=1):
    wait_buffer, replay_buffer = buffers
    replay_buffer.push(*sample)
    return wait_buffer
    

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
    

def train(num_frames, DQN, multi_step=1):
    
    def maxQ(model, state):
        with torch.no_grad():
            state = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(device)
            q_value = model.forward(state)
            max_q_value = q_value.max(1).values.item()
        return max_q_value
    replay_buffer = ReplayBuffer(replay_buffer_size)
    
    wait_buffer = []
    current_model = DQN(input_n, discrete_action_n, h_size=h_size).to(device)
    target_model = None
    if USE_TARGET_NET:
        target_model = DQN(input_n, discrete_action_n, h_size=h_size).to(device)
        update_target(current_model, target_model)
    optimizer = optim.Adam(current_model.parameters(), lr=learning_rate)
    max_Qs = []
    now_Q = 0
    all_rewards = []
    episode_reward = 0
    episode = 0
    state = env.reset()
    env.seed(SEED)
    # state, _ = env.reset(seed=SEED)
    for frame_idx in tqdm(range(1, num_frames + 1)):
        t_episode_start = time.process_time()
        epsilon = epsilon_by_frame(frame_idx)
        action = current_model.act(state, epsilon, device, discrete_action_n)
        now_Q = maxQ(current_model, state)
        max_Qs.append(now_Q)
        if not isinstance(env.action_space, gym.spaces.discrete.Discrete):
            next_state, reward, done, _ = env.step(np.array([faction(action)]))
        else:
            next_state, reward, done, _ = env.step(action)
        wait_buffer = push_buffer((wait_buffer, replay_buffer), (state, action, reward, next_state, done), step=multi_step)
        episode_reward += reward
        state = next_state
        if done:
            state = env.reset()
            episode_reward = 0
            episode += 1
        if frame_idx % eval_step == 0:
            r = evaluation(current_model)
            all_rewards.append(r)
        if len(replay_buffer) >= replay_initial:
            learn(batch_size, current_model, target_model, replay_buffer, optimizer, step=multi_step)
        if frame_idx % update_target_step == 0:
            if USE_TARGET_NET:
                update_target(current_model, target_model)
    print(target_model)
    torch.save(target_model.state_dict(), './weights/target_model_{}_{}.pt'.format(ENV_NAME, SEED))

    return max_Qs, all_rewards


if __name__ == '__main__':
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

    # algo_list = ['DQN', 'DQN-wo-target', 'DQN-wo-buffer', 'DoubleDQN']
    algo_list = ['DQN']
    return_list = []
    maxQ_list = []
    for ALGO  in algo_list:
        if ALGO == 'DQN': # vanilla DQN
            USE_REPLAY_BUFFER = True
            USE_TARGET_NET = True
            USE_DOUBLE_Q = False
        elif ALGO == 'DQN-wo-target': # DQN w/o using target net
            USE_REPLAY_BUFFER = True
            USE_TARGET_NET = False
            USE_DOUBLE_Q = False
        elif ALGO == 'DQN-wo-buffer': # DQN w/o using replay buffer
            USE_REPLAY_BUFFER = False
            USE_TARGET_NET = True
            USE_DOUBLE_Q = False
        elif ALGO == 'DoubleDQN': # double DQN
            USE_REPLAY_BUFFER = True
            USE_TARGET_NET = True
            USE_DOUBLE_Q = True

        replay_initial = 1000 if USE_REPLAY_BUFFER else 1
        replay_buffer_size = 3000 if USE_REPLAY_BUFFER else 1
        batch_size = 64 if USE_REPLAY_BUFFER else 1

        maxQ_value, episode_return = train(num_frames, DQN=DQN)
        maxQ_list.append(maxQ_value)
        return_list.append(episode_return)

    compare_result(return_list, algo_list, filename = './fig/return_comparison_{}_{}'.format(ENV_NAME, SEED))
    # compare_result([maxQ_list[0],maxQ_list[1]] , [algo_list[0], algo_list[1]], use_max=True, filename = 'return_comparison_{}_{}'.format(algo_list[0], algo_list[1]))
    # compare_result([maxQ_list[0],maxQ_list[2]] , [algo_list[0], algo_list[2]], use_max=True, filename = 'return_comparison_{}_{}'.format(algo_list[0], algo_list[2]))
    # compare_result([maxQ_list[0],maxQ_list[3]] , [algo_list[0], algo_list[3]], use_max=True, filename = 'return_comparison_{}_{}'.format(algo_list[0], algo_list[3]))
    # compare_result([maxQ_list[0],maxQ_list[3]] , [algo_list[0], algo_list[3]], use_max=True, ylabel_name="maxQ", filename = 'maxQ_comparison_{}_{}'.format(algo_list[0], algo_list[3]))
