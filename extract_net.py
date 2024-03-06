'''
This is the file to extract the net from the standard model.

@Author: Xiangyun Rao
@Date: 2023.7.13
'''
import math

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from agent import DQN, ExtractNet
from buffer import ReplayBuffer
from tool import plot_loss

def extract_from_current_model(extractor, mybuffer, batch_size, device, epochs, optimizer):
    buffer_size = len(mybuffer)
    time_step = buffer_size // batch_size
    counter = 0
    losses = []
    x = []
    for epoch in range(epochs):
        counter += 1
        state, action, reward, next_state, done = mybuffer.sample(batch_size)
        with torch.no_grad():
            state = torch.FloatTensor(np.float32(state)).to(device)
            next_state = torch.FloatTensor(np.float32(next_state)).to(device)
            action = torch.LongTensor(action).to(device)
            reward = torch.FloatTensor(reward).to(device)
            done = torch.FloatTensor(done).to(device)
        q_values = extractor(state)
        next_q_values = extractor(next_state)
        # # next_q_values_target = current_model(next_state)
        # get q value
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # get next q value
        action = torch.max(next_q_values, 1)[1].unsqueeze(1)
        next_q_value = next_q_values.gather(1, action).squeeze(1)
        gamma = 0.98
        # calculate expected q value according to bellman equation
        # next_q_values = current_model(next_state)
        # action = torch.max(next_q_values, 1)[1].unsqueeze(1)
        # next_q_value = next_q_values.gather(1, action).squeeze(1)
        # gamma = 0.98
        # calculate expected q value according to bellman equation
        expected_q_value = reward + gamma * next_q_value * (1 - done)
        # get loss
        loss = F.mse_loss(q_value, expected_q_value)
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # check whether the epoch is finished
        if counter >= time_step:
            losses.append(loss.item())
            x.append(epoch)
            counter = 0
            print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))

    plot_loss(x, losses, 'Extractor_Loss')


if __name__ == '__main__':

    replay_buffer_size = 3000
    batch_size = 32
    learning_rate = 0.001
    Epochs = 10000
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 会出现state无法转为numpy的问题
    device = torch.device("cpu")
    SEED = 9
    ENV_NAME = "Pendulum-v0"

    env = gym.make(ENV_NAME)
    input_n = env.observation_space.shape[0]
    discrete_action_n = 11
    h_size = 24
    mybuffer = ReplayBuffer(replay_buffer_size)
    mybuffer.load('buffer_{}_{}.pkl'.format(ENV_NAME, SEED))

    current_model = DQN(input_n, discrete_action_n, h_size=h_size).to(device)
    extractor = ExtractNet(input_n, discrete_action_n).to(device)
    model_dict = torch.load('./weights/target_model_{}_{}.pt'.format(ENV_NAME, SEED))
    # rename 'fc.2.weight' to 'fc.1.weight' etc.
    model_dict_origin = model_dict.copy()
    for key in list(model_dict.keys()):
        model_dict[key.replace('fc.2', 'fc.1')] = model_dict.pop(key)
    # mutiply the weight and bias of the first layer by  \frac{e-1}{e+1}
    k = (math.e - 1) / (math.e + 1)
    model_dict['fc.1.weight'] = model_dict['fc.1.weight'] * k
    model_dict['fc.1.bias'] = model_dict['fc.1.bias'] * k
    # load the model
    current_model.load_state_dict(model_dict_origin)
    extractor.load_state_dict(model_dict)
    # set not to train
    current_model.eval()
    optimizer = optim.Adam(extractor.parameters(), lr=learning_rate)

    # extract the net
    extract_from_current_model(extractor, mybuffer, batch_size, device, Epochs, optimizer)
    torch.save(extractor.state_dict(), './weights/ExtractNet_{}_{}.pt'.format(ENV_NAME, SEED))
