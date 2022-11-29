# -*- coding:utf-8 -*-
import argparse
import os
import random
import time

import gym
import numpy as np
import torch

from dqn import parse_args, make_env, QNetwork

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = make_env(args.env_id, args.seed)
    
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(torch.load("./output/dqn.pth")) # Load parameters from file.

    obs = envs.reset()
    dones = False

    record_video = 0 # use a flag to set whether recording a video, you can modify it.
    if record_video:
        envs = gym.wrappers.RecordVideo(envs, video_folder = "./video", name_prefix = "dqn")
    else:
        envs.render() 
    while not dones:
        q_values = target_network(torch.Tensor(obs).to(device))
        actions = torch.argmax(q_values, dim=0).cpu().numpy()
        obs, rewards, dones, infos = envs.step(actions)
        if not record_video:
            envs.render()
    envs.close()