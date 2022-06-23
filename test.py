#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import torch
from torchvision import datasets, transforms
from network import CNN
from dataset import subset_MNIST

from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

##
## test main function
##
def test(args, model, device):
    model.eval()

    nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # nums = [3]
    testset = subset_MNIST(root='~/.pytorch', nums=nums, train=True, download=True, transform=transforms.ToTensor())

    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)
    data, labels = testloader.__iter__().next()

    pred = model.forward(data).to('cpu').detach().numpy().copy()

    for j, (d, p) in enumerate(zip(data, pred)):

        image = d.reshape(28, 28) * 255
        image = image.detach().numpy()
        plt.cla()
        plt.imshow(image, cmap="gray")
        for i in range(10):
            if p[i] >= 0.00001:
                plt.text(29, 1 + i * 1.4, "%d: %0.5f"%(i, p[i]), color="red")
            else:
                plt.text(29, 1 + i * 1.4, "%d: %0.5f"%(i, p[i]), color="gray")
                
        plt.axis("off")

        # print("result/%03d.png"%j)
        # plt.savefig("result/%03d.png"%j)
        
        plt.pause(5)



##
## main function
##
def main():
# Testing settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', default="result/mnist.pt")

    args = parser.parse_args()

    if args.model == "":
        print("Usage: test.py --model modelname.pt")
        exit()

    device = torch.device("cpu")

    model = CNN().to(device)

    modelname = args.model

    model.load_state_dict(torch.load(modelname))
    
    #
    with torch.no_grad():
        test(args, model, device)

if __name__ == '__main__':
    main()

# import pdb; pdb.set_trace()

### Local Variables: ###
### truncate-lines:t ###
### End: ###

