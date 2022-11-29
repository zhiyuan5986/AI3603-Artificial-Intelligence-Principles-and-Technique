# -*- coding:utf-8 -*-
import math, os, time, sys
import numpy as np
import gym
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

class SarsaAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions, epsilon, alpha, gamma):
        """initialize the agent. Maybe more function inputs are needed."""
        # all_actions is a numpy array of all the actions, np.array([0,1,2,3])
        self.all_actions = all_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q_s_a = {} # self.Q_s_a is a dict with the form of key as a tuple of s and a, i.e. key: (state, action)

    def get_Q(self,state, action):
        self.Q_s_a.setdefault((state, action), 0)
        return self.Q_s_a[(state, action)]

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        # The observation is simply the current position encoded as flattened index. (From https://www.gymlibrary.dev/environments/toy_text/cliff_walking/)
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.all_actions)
        else:
            Q = [self.get_Q(observation, action) for action in self.all_actions]
            action = self.all_actions[np.argmax(Q)]
        return action
    
    def learn(self, state, action, next_state, next_action, reward):
        """learn from experience"""
        # next_action = self.choose_action(next_state)
        next_Q_s_a = self.get_Q(next_state, next_action)
        self.Q_s_a[(state,action)] = (1-self.alpha)*self.get_Q(state, action) + self.alpha * (reward+self.gamma*next_Q_s_a)
    
    def your_function(self, params):
        """You can add other functions as you wish."""
        return None

    ##### END CODING HERE #####


class QLearningAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions, epsilon, alpha, gamma):
        """initialize the agent. Maybe more function inputs are needed."""
        # all_actions is a numpy array of all the actions, np.array([0,1,2,3])
        self.all_actions = all_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q_s_a = {} # self.Q_s_a is a dict with the form of key as a tuple of s and a, i.e. key: (state, action)

    def get_Q(self,state, action):
        self.Q_s_a.setdefault((state, action), 0)
        return self.Q_s_a[(state, action)]

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        # The observation is simply the current position encoded as flattened index. (From https://www.gymlibrary.dev/environments/toy_text/cliff_walking/)
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.all_actions)
        else:
            Q = [self.get_Q(observation, action) for action in self.all_actions]
            action = self.all_actions[np.argmax(Q)]
        return action
    
    def learn(self, state, action, next_state, reward):
        """learn from experience"""
        # next_action = self.choose_action(next_state)
        max_next_Q_s_a = max(self.get_Q(next_state, action) for action in self.all_actions)
        self.Q_s_a[(state,action)] = (1-self.alpha)*self.get_Q(state, action) + self.alpha * (reward+self.gamma*max_next_Q_s_a)
    
    def your_function(self, params):
        """You can add other functions as you wish."""
        return None

    ##### END CODING HERE #####
