# -*- coding:utf-8 -*-
# Train Sarsa in cliff-walking environment
import math, os, time, sys
import numpy as np
import random
import gym
from agent import SarsaAgent
from matplotlib import pyplot as plt
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

##### END CODING HERE #####

# construct the environment
env = gym.make("CliffWalking-v0")
# get the size of action space 
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
# set random seed and make the result reproducible
RANDOM_SEED = 0
env.seed(RANDOM_SEED)
random.seed(RANDOM_SEED) 
np.random.seed(RANDOM_SEED) 

####### START CODING HERE #######

# construct the intelligent agent.
alpha = 1.
gamma = .9
epsilon = 1.
agent = SarsaAgent(all_actions, alpha, gamma, epsilon)
reward_list = [0]*1000
epsilon_list = [0]*1000
path = []

# start training
for episode in range(1000):
    # record the reward in an episode
    episode_reward = 0
    # record epsilon
    epsilon_list[episode] = agent.epsilon
    # reset env
    s = env.reset()
    a = agent.choose_action(s)
    # render env. You can remove all render() to turn off the GUI to accelerate training.
    # env.render()
    # agent interacts with the environment
    for iter in range(500):
        # choose an action
        s_, r, isdone, info = env.step(a)        
        a_ = agent.choose_action(s)
        # env.render()
        # update the episode reward
        episode_reward += r
        print(f"{s} {a} {s_} {r} {isdone}")
        # agent learns from experience
        agent.learn(s,a,s_,a_,r)
        s,a = s_,a_
        if isdone:
            time.sleep(0.1)
            break
    print('episode:', episode, 'episode_reward:', episode_reward, 'alpha:', agent.alpha, 'epsilon:', agent.epsilon) 
    reward_list[episode] = episode_reward 

    agent.alpha -= 0.0009
    agent.epsilon *= 0.99
print('\ntraining over\n')   

# close the render window after training.

# test
s = env.reset()
path.append(np.unravel_index(s,(4,12)))
agent.epsilon = 0
reward = 0

record_video = 0 # use a flag to set whether recording a video, you can modify it.
if record_video:
    env = gym.wrappers.RecordVideo(env, video_folder = "./video", name_prefix = "sarsa")
else:
    env.render()
while True:
    a = agent.choose_action(s)
    s_, r, isdone, info = env.step(a)
    reward += r
    if not record_video:
        env.render()
    s = s_
    path.append(np.unravel_index(s,(4,12)))
    if isdone:
        break
env.close()
print(f"reward: {reward}")

# plot
## 1. reward list
if not os.path.exists("./output"):
    os.mkdir("./output")
plt.figure()
plt.plot(reward_list)
plt.savefig("./output/cliff_walk_sarsa_reward.pdf")

## 2. epsilon list
plt.figure()
plt.plot(epsilon_list)
plt.savefig("./output/cliff_walk_sarsa_epsilon.pdf")

## 3. path
print(path)
path = np.array(path)
plt.figure(figsize=(6,3))

plt.xticks([i for i in range(12)])
plt.yticks([i for i in range(4)])
plt.ylim(-0.5,3.5)
ax = plt.gca()                       # Get the current axis information
ax.xaxis.set_ticks_position('top')   # Change x-axis to up
ax.invert_yaxis()                    # Invert y-axis
ax.set_aspect('equal', adjustable='box')

plt.plot(path[:,1],path[:,0])
plt.plot(path[0][1],path[0][0],"g^")
plt.plot(path[-1][1],path[-1][0],"r*")
plt.title(f"Reward = {reward}")
plt.savefig("./output/cliff_walk_sarsa_path.pdf")

####### END CODING HERE #######


