"""
This is the file to compare the real parameters and the identified parameters.

@Author: Xiangyun Rao
@Date: 2024.4.16
"""

import torch
import torch.nn as nn
import pickle

from agent import ExtractNet


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

def compare_params(W, B, W_, B_):

    print("The real parameters are: ")
    print("W: ", W)
    print("B: ", B)
    print("The identified parameters are: ")
    print("W_: ", W_)
    print("B_: ", B_)

    print("The difference between real parameters and identified parameters are: ")
    print("W_diff: ", abs(W - W_))
    print("B_diff: ", abs(B - B_))

    return W - W_, B - B_


if __name__ == "__main__":

    input_dim = 3
    output_dim = 11
    device = "cpu"

    extractor = ExtractNet(input_dim, output_dim).to(device)

    # load the identified parameters
    with open('./results/WBresults_Pendulum-v0_9.pt', 'rb') as f:
        W, B = pickle.load(f)

    # load the real parameters
    W_, B_ = get_parameters(extractor)

    # compare the real parameters and the identified parameters
    delta_W, delta_B = compare_params(W, B, W_, B_)

