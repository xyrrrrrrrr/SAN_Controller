'''
This is the file to train the SAN controller.

@Author: Xiangyun Rao
@Date: 2023.7.30
'''
import math
import gym
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent import ExtractNet, SANController
from buffer import ReplayBuffer
from tool import plot_loss
from tqdm import tqdm


# Reconstruction + KL divergence losses summed over all elements and batch    
class MyLoss(nn.Module):
    '''
    This is the loss function for SAN controller.
    @param: u: original state
    @param: recon_u: reconstructed state
    @param: expected_u: expected state
    @param: Q: Q value
    @param: Q_0: expected Q value

    @return: loss
    '''
    def __init__(self):
        super(MyLoss, self).__init__()
        self.kld_loss = nn.KLDivLoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, u, recon_u, expected_u, Q, Q_0):
        # kld
        kld_loss = self.kld_loss(u, recon_u)
        if math.isnan(kld_loss.item()):
            kld_loss = torch.FloatTensor([0.0])
        if kld_loss.item() > 1:
            kld_loss = torch.FloatTensor([1.0])
        Q_loss = self.mse_loss(Q, Q_0)
        # expected u loss
        main_loss = self.mse_loss(u, expected_u)
    
        return 0.5 * Q_loss + 0.05 * kld_loss + 0.45 * main_loss


def train_san(san, extractor, my_buffer, batch_size, loss_func, device, epochs, optimizer, grad_theshold)->None:
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(san.parameters(), grad_theshold)
    # Get Weight and Bias
    W_1 = extractor.fc[0].weight
    b_1 = extractor.fc[0].bias
    W_2 = extractor.fc[1].weight
    b_2 = extractor.fc[1].bias

    # Calculate W and b, W = W_2 * W_1, b = W_2 * b_1 + b_2, y = Wx + b
    W = torch.matmul(W_2, W_1)
    b = torch.matmul(W_2, b_1) + b_2

    # start training
    buffer_size = len(my_buffer)
    time_step = buffer_size // batch_size
    counter = 0
    x = []
    losses = []
    for epoch in tqdm(range(epochs)):
        counter += 1
        state_0, action, reward, next_state, done = my_buffer.sample(1)
        with torch.no_grad():
            state_0 = torch.FloatTensor(np.float32(state_0)).to(device)
            next_state = torch.FloatTensor(np.float32(next_state)).to(device)
            action = torch.LongTensor(action).to(device)
            reward = torch.FloatTensor(reward).to(device)
            done = torch.FloatTensor(done).to(device)
        # Calculate expected state
        # 0. mutiply state_0 with state_0.T
        target = torch.matmul(state_0.t(), state_0)    
        # 1. calculate state_0's svd decomposition
        V_2, S, V_1 = torch.linalg.svd(target)
        # V_2, S, V_1 = torch.svd(target) 已被torch弃用
        # print(V_2, S, V_1)
        # print(V_1)
        # 2. Obtain reduced state
        r = target.shape[0]
        target_reduced = torch.matmul(V_1, target)
        # 3. Get P_r from Riccati equation 
        # A_{r}^{T}P_{r}A_{r}-A_{r}^{T}P_{r}V_{1}^{T}(I+V_{1}P_{r}V_{1}^{T})^{-1}V_{1}P_{r}A_{r}-P_{r}+C_{r}^{T}C_{r}=0
        A_r = V_1.t() * V_2
        B_r = V_1.t()
        C_r = W @ V_2
        A_rn = A_r.cpu().numpy() if A_r.is_cuda else A_r.numpy()
        B_rn = B_r.cpu().numpy() if B_r.is_cuda else B_r.numpy()
        C_rn = C_r.cpu().detach().numpy() if C_r.is_cuda else C_r.detach().numpy()
        P_r = scipy.linalg.solve_continuous_are(A_rn, B_rn, C_rn.T @ C_rn, np.eye(r))
        # print('P_r: ', P_r)
        P_r = torch.FloatTensor(P_r).to(device)
        # 4. Get L_r which is given by L_r = R^{-1} B_r^{T} P_r
        R = torch.eye(r).to(device)
        L_r = torch.matmul(torch.inverse(R), B_r.t()) @ P_r
        # 5. Get expected target which is given by expected_target = - L_r * target_reduced
        expected_target = - L_r @ target_reduced
        # print('expected_target: ', expected_target)
        # 6. Get expected state, expected_state @ expected_state.T = expected_target
        diag = torch.diag(expected_target)
        bias = torch.sqrt(torch.abs(diag))
        normalized_bias = bias / torch.sum(bias)
        normalized_bias = torch.where(normalized_bias > 0.1, 0, normalized_bias)
        expected_state = state_0 + normalized_bias

        # Add disturbance
        state = state_0 + san(state_0)
        q_values = extractor(state.detach())
        next_q_values = extractor(next_state)
        # # next_q_values_target = current_model(next_state)
        # get q value
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # get next q value
        action = torch.max(next_q_values, 1)[1].unsqueeze(1)
        next_q_value = next_q_values.gather(1, action).squeeze(1)
        gamma = 0.98
        # calculate expected q value according to bellman equation
        expected_q_value = reward + gamma * next_q_value * (1 - done)
        # get loss
        loss = loss_func(state_0, state, expected_state, q_value, expected_q_value)
        # print('Loss: ', loss.item())
        # losses.append(loss.item())
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # check whether the epoch is finished
        if counter >= time_step:
            counter = 0
            losses.append(loss.item())
            x.append(epoch)
    plot_loss(x, losses, 'SAN_Loss')
    
if __name__ == '__main__':
    # Hyper parameters
    replay_buffer_size = 3000
    batch_size = 32
    learning_rate = 0.001
    Epochs = 10000
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    SEED = 9
    ENV_NAME = "Pendulum-v0"

    env = gym.make(ENV_NAME)
    input_n = env.observation_space.shape[0]
    discrete_action_n = 11
    h_size = 24
    my_buffer = ReplayBuffer(replay_buffer_size)
    my_buffer.load('buffer_{}_{}.pkl'.format(ENV_NAME, SEED))

    # Create model
    extractor = ExtractNet(input_n, discrete_action_n, h_size=h_size).to(device)
    san = SANController(input_n=input_n, output_n=input_n).to(device)
    
    # Load Extractor
    extractor.load_state_dict(torch.load('./weights/ExtractNet_Pendulum-v0_9.pt'))
    extractor.eval()

    # Set optimizer
    optimizer = optim.Adam(san.parameters(), lr=learning_rate)

    # Set loss function
    loss_func = MyLoss()

    # Set gradient threshold
    grad_theshold = 0.5

    # Train
    train_san(san, extractor, my_buffer, batch_size, loss_func, device, Epochs, optimizer, grad_theshold)

    # Save model
    torch.save(san.state_dict(), './weights/SANController_{}_{}.pt'.format(ENV_NAME, SEED))
    

    

