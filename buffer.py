'''
This is the file to define the replay buffer.

@Author: Xiangyun Rao
@Date: 2023.7.13
'''
import pickle
import random
from collections import deque

import numpy as np


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        '''
        Args: state (ndarray (3,)), action (int), reward (float), next_state (ndarray (3,)), done (bool)

        No return
        '''
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        '''
        Args: batch_size (int)

        Required return: a batch of states (ndarray, (batch_size, state_dimension)), a batch of actions (list or tuple, length=batch_size),
        a batch of rewards (list or tuple, length=batch_size), a batch of next-states (ndarray, (batch_size, state_dimension)),
        a batch of done flags (list or tuple, length=batch_size)
        '''
        # 初始化状态、动作、奖励、下一个状态、是否结束标志
        batch_state = np.zeros((batch_size, 3))
        batch_action = []
        batch_reward = []
        batch_next_state = np.zeros((batch_size, 3))
        batch_done = []
        # 按照batch_size的大小从buffer中随机抽取样本
        sample = random.sample(self.buffer, batch_size)
        # 将抽取的样本分别放入对应的列表中
        for i in range(batch_size):
            batch_state[i] = sample[i][0]
            batch_action.append(sample[i][1])
            batch_reward.append(sample[i][2])
            batch_next_state[i] = sample[i][3]
            batch_done.append(sample[i][4])

        return batch_state, batch_action, batch_reward, batch_next_state, batch_done

    def load(self, file_name):
        '''
        Args: file_name (str)

        used to load the replay buffer from a file
        '''
        file_name = "buffer/" + file_name

        with open(file_name, 'rb') as f:
            self.buffer = pickle.load(f)

    def __len__(self):
        return len(self.buffer)
