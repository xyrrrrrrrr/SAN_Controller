'''
This is the file to define the agent.

@Author: Xiangyun Rao
@Date: 2023.7.13
'''
# 系统级库在开头
import random
# 第三方库在后，按照字母顺序排列
import gym
import numpy as np
import torch
import torch.nn as nn
# import结束空两行

ENV = gym.make('Pendulum-v0')
STATE_LOW = ENV.observation_space.low
STATE_HIGH = ENV.observation_space.high
# 全局变量后空两行

def normalize_state(state):
    return (state - STATE_LOW) / (STATE_HIGH - STATE_LOW)
# 函数与类定义后空两行

class DQNBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

    def act(self, state, epsilon, device, discrete_action_n):
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(
                    np.float32(state)
                ).unsqueeze(0).to(device)
                q_value = self.forward(state)
                action = q_value.max(1).indices.item()
        else:
            action = random.randrange(discrete_action_n)
        return action

    
class DQN(DQNBase):
    def __init__(self, input_n, num_actions, h_size=24):
        super(DQN, self).__init__()

        self.input_n = input_n
        self.num_actions = num_actions

        self.fc = nn.Sequential(
            nn.Linear(self.input_n, h_size),
            # nn.ReLU(),
            nn.Tanh(),
            # nn.Linear(h_size, h_size),
            # # nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(h_size, self.num_actions)
        )

    def forward(self, x):
        normalize_state(x)
        x = self.fc(x)
        return x
    

class ExtractNet(DQNBase):
    def __init__(self, input_n, num_actions, h_size=24):
        super(ExtractNet, self).__init__()

        self.input_n = input_n
        self.num_actions = num_actions

        self.fc = nn.Sequential(
            nn.Linear(self.input_n, h_size),
            nn.Linear(h_size, self.num_actions)
        )
    
    def forward(self, x):
        normalize_state(x)
        x = self.fc(x)
        return x
 

class SANController(nn.Module):
    '''
    this is a encoder-decoder network
    '''
    def __init__(self, input_n, output_n):
        super(SANController, self).__init__()
        self.input_n = input_n
        self.output_n = output_n

        self.encoder = nn.Sequential(
            nn.Linear(self.input_n, 24),
            nn.Tanh(),
            nn.Linear(24, 12),
            nn.Tanh(),
            nn.Linear(12, 6),
            nn.Tanh(),
            nn.Linear(6, 3),
            nn.Tanh()
        )
        self.mean = nn.Linear(3, 3)
        self.log_std = nn.Linear(3, 3)

        self.decoder = nn.Sequential(
            nn.Linear(3, 6),
            nn.Tanh(),
            nn.Linear(6, 12),
            nn.Tanh(),
            nn.Linear(12, 24),
            nn.Tanh(),
            nn.Linear(24, self.output_n)
        )

    def forward(self, x):
        x = self.encoder(x)
        m = self.mean(x)
        s = self.log_std(x)
        z = m + s.exp() * torch.randn_like(s)
        x = self.decoder(z)
        
        return x
