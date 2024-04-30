'''
This is the file to define the agent.

@Author: Xiangyun Rao
@Date: 2024.3.6
'''

import matplotlib.pyplot as plt
import numpy as np

def plot_loss(x, losses, title):
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title(title)
    plt.plot(x, losses, linewidth = 1)
    plt.savefig('./fig/' + title + '.png')


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def smooth_curve(y, smooth):
    r = smooth
    length = int(np.prod(y.shape))
    for i in range(length):
        if i > 0:
            if (not np.isinf(y[i - 1])) and (not np.isnan(y[i - 1])):
                y[i] = y[i - 1] * r + y[i] * (1 - r)
    return y


def moving_average(y, x=None, total_steps=100, smooth=0.9, move_max=False):
    if isinstance(y, list):
        y = np.array(y)
    length = int(np.prod(y.shape))
    if x is None:
        x = list(range(1, length+1))
    if isinstance(x, list):
        x = np.array(x)
    if length > total_steps:
        block_size = length//total_steps
        select_list = list(range(0, length, block_size))
        select_list = select_list[:-1]
        y = y[:len(select_list) * block_size].reshape(-1, block_size)
        if move_max:
            y = np.max(y, -1)
        else:
            y = np.mean(y, -1)
        x = x[select_list]
    y = smooth_curve(y, smooth)
    return y, x
        

def plot_maxQ(max_Qs):
    plt.clf()
    y, x = moving_average(max_Qs)
    plt.xlabel("frame")
    plt.ylabel("Q value")
    plt.plot(x, y)
    plt.show()
    plt.savefig('MaxQ.png')


def plot_reward(all_rewards):
    plt.clf()
    y, x = moving_average(all_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(x, y)
    plt.savefig('reward.png')
    plt.show()    


def compare_result(rewards, labels, use_max = False, smooth = 0.9, eval_step = 200, ylabel_name = "Reward", filename=None):
    plt.clf()
    plt.ylabel(ylabel_name)
    plt.xlabel("Frame")
    for rs in rewards:
        if not use_max:
            x1 = list(range(0, len(rs)*eval_step, eval_step))
        else:
            x1 = None
        reward1, x1 = moving_average(rs, x1, move_max=use_max, smooth=smooth)
        plt.plot(x1, reward1)
    plt.legend(labels)
    plt.savefig('{}.png'.format(filename))
    plt.show()

def plot_reward(rewards, title, y_lim = None, fig_size = (10, 5)):
    plt.figure(figsize = fig_size)
    # 设置y轴的范围
    if y_lim:
        plt.ylim(y_lim, 0)
    plt.plot(rewards)
    plt.title(title)
    plt.savefig("./fig/" + title + '.png')
    plt.close()

