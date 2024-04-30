# -*- coding: utf-8 -*-
"""
This is the file to prove the protection of the system.

@Author: Xiangyun Rao
@Date: 2024.3.20
"""
#导入相应科学计算的包
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gym
import pickle
from agent import ExtractNet, SANController, IdentifyNet

#显示中文字体
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

# 获取y=WX+B的参数
def get_parameters(extractor):

    W_1 = extractor.fc[0].weight
    b_1 = extractor.fc[0].bias
    W_2 = extractor.fc[1].weight
    b_2 = extractor.fc[1].bias
    # calculate the parameters
    W = torch.mm(W_2, W_1)
    B = W_2 @ b_1 + b_2
    print("W的值是：", W, "B的值是：", B)

    return W, B

# 产生一组L个N(0,1)的高斯分布的随机噪声
def generate_gaussian_noise():

    global mu, sigma, L, output_dim
    noise = [torch.tensor(np.random.normal(mu, sigma, output_dim), dtype=torch.float32) for _ in range(L)]
    plt.figure(num=1)
    plt.plot(noise[0].numpy(), label='1st dim')
    plt.plot(noise[1].numpy(), label='2nd dim')
    plt.plot(noise[2].numpy(), label='3rd dim')
    plt.title("高斯分布随机噪声")
    plt.savefig('./fig/高斯分布随机噪声.png')

    return noise

# 通过SANController和Extractor获取输出
def get_output(extractor, SANController, origin_input):
    
    origin_input = torch.tensor(origin_input, dtype=torch.float32).detach()
    real_input = origin_input + SANController(origin_input).detach()
    output = extractor(real_input).detach()

    return output

# 收集y和u，u是模拟输入，y是的真实输出
def collect_y_u(extractor, SANController):

    global input_dim, output_dim, L, random_scale
    #定义初始值
    u = []
    y = []    
    for _ in range(L):
        # 产生一个模拟输入
        current_u = [np.random.uniform(random_scale[j][0], random_scale[j][1]) for j in range(input_dim)]
        current_u = torch.tensor(current_u, dtype=torch.float32)
        # 获取真实输出
        current_y = get_output(extractor, SANController, current_u)
        u.append(current_u)
        y.append(current_y)   

    return y, u

def system_identify_with_least_squares(y, u, noise, identify_net, epochs = 300, lr = 0.0001):
    """
    通过最小二乘辨识系统
    
    Args:
        y: 观测输出, shape=(L, output_dim)
        u: 模拟输入, shape=(L, input_dim)
        noise: 高斯分布随机噪声

    Returns:
        W: 系统参数W, shape=(input_dim, output_dim)
        B: 系统参数B, shape=(output_dim)
    """
    global L

    order = [i - 1 for i in range(L)]
    losses = []
    z = y + noise
    for i in range(epochs):
        # 随机打乱顺序
        loss = 0.
        np.random.shuffle(order)
        for j in range(L):
            # 获取当前的观测输出和模拟输入
            current_z = z[order[j]]
            current_u = u[order[j]]
            # 获取当前的输出
            outputs = identify_net(current_u)
            # 计算残差
            residual = current_z - outputs
            # 更新参数
            identify_net.zero_grad()
            loss_ = torch.sum(residual ** 2)
            loss_.backward()
            loss += loss_    
            with torch.no_grad():
                for param in identify_net.parameters():
                    param -= lr * param.grad
        if i % 10 == 0:
            losses.append(loss.item()/L)
            print("epoch:{}, loss:{}".format(i, loss.item()/L))
    W, B = identify_net.get_params()
    print("辨识得到的W是：", W, "辨识得到的B是：", B)
    pickle.dump([W, B], open('./results/WBresults_Pendulum-v0_9.pt', 'wb'))
    plt.figure(num=3)
    plt.plot(losses)
    plt.title("最小二乘法的损失函数")
    plt.savefig('./fig/最小二乘法的损失函数.png')

    return W, B

    

if __name__ == '__main__':
    #设定初始参数
    L = 1000
    mu = 0
    sigma = 0.05
    input_dim = 3
    output_dim = 11
    random_scale = [[-1,1], [-1,1], [-8,8]]
    device = torch.device("cpu")
    # 引入agent
    extractor = ExtractNet(input_dim, output_dim).to(device)
    san = SANController(input_dim, input_dim).to(device)
    identify_net = IdentifyNet(input_dim, output_dim).to(device)
    extractor.load_state_dict(torch.load('./weights/ExtractNet_Pendulum-v0_9.pt'))
    san.load_state_dict(torch.load('./weights/SANController_Pendulum-v0_9.pt'))
    extractor.eval()
    san.eval()
    # 获取参数
    W, B = get_parameters(extractor)
    # 产生高斯分布的随机噪声
    noise = generate_gaussian_noise()
    # 收集y和u，u是模拟输入，y是观测输出
    y, u = collect_y_u(extractor, san)
    # 进行辨识
    W_, B_ = system_identify_with_least_squares(y, u, noise, identify_net)
    