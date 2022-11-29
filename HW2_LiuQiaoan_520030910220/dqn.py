# -*- coding:utf-8 -*-
import argparse
import os
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    """parse arguments. You can add other arguments if needed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-3,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1.,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.1,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    args = parser.parse_args()
    args.env_id = "LunarLander-v2"
    return args

def make_env(env_id, seed):
    """construct the gym environment"""
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

class QNetwork(nn.Module):
    """comments: 
        This class defines the Neural Network used as a function of $Q(s,a)$, 
        which takes $s$ as input and output all $Q(s,a)$ for each action as an array.
                 
        The network first use a fully-connected layer, with input size as the size of state space, and output 120 features.
        Second, a ReLU layer is used to find nonlinear features.
        Third, another fully-connected layer.
        Forth, another ReLU layer.
        Finally, size of output array is equal to size of action space. The output array indicates the probability to take each action.
    """
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """comments: 
        This function takes `start_e`, `end_e`, means the $\epsilon$ at the beginning and at the end, respectively. 
        This function linearly decreases the value of $\epsilon$ at each time step.
    """
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    
    """parse the arguments"""
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    """we utilize tensorboard yo log the training process"""
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    """comments: 
        1. Set random seed as `arg.seed`.
        2. Set numpy's random seed as `arg.seed`.
        3. Set torch's random seed as `arg.seed`.
        4. After setting torch's random seed to be deterministic, we need to set cuda's random seed to be `arg.seed`,
           otherwise, if we use cuda (GPU), it will use different random seed each time we run the Python script.
        5. Set `device` to be "cuda" if cuda is available, i.e. GPU can be used. Else, we use CPU.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    """comments: 
        Use `make_env` function to construct an env 
        In `make_env` function:
            1. `env_id` means which toy game to play, use `gym.make` to make an env.
            2. `gym.wrappers.RecordEpisodeStatistics` ensures that we can get `infos` after each step of action, 
               which keep track of cumulative rewards and episode lengths.
               From https://github.com/openai/gym/blob/master/gym/wrappers/record_episode_statistics.py
            3. Set each random seed to be `seed`.
    """
    envs = make_env(args.env_id, args.seed)

    """comments: 
        1. Construct a Neural Network and deploy it to `device`.
        2. Use Adaptive Moment Estimation Algorithm to optimize the parameters of the NN, with a learning rate as `args.learning_rate`.
        3. Construct a target NN. Target NN is used to store the Q-table, which is needed when calculate $\max Q(s,a)$.
           NOTE: This ensures target NN will not change frequently, we can use it with a batch of data (combined with Replay Buffer).
        4. Load the trained parameters to target NN.
    """
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    """comments: 
        Replay Buffer is used in DQN, which can solve two problems in Q-learning:
            1. Correlated data 
            2. Non-stationary distributions
        First, Since in each episode, the data across each step is high correlated. 
        If we just update after each step, there will be relatively high covariance.
        Second, we can use a batch of data to train at a time, other than use each $(s,a)$, 
        which can make the system more stationary.
    """
    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        handle_timeout_termination=False,
    )

    """comments: 
        Get corrent observation (the current state).
    """
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        
        """comments: 
            Get $\epsilon$ as a linearly decreased number at each time step.
        """
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        
        """comments: 
            We randomly select an action to explore with a probability of $\epsilon$
            Else, we get the best action according to the output of `q_network`
        """
        if random.random() < epsilon:
            actions = envs.action_space.sample()
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=0).cpu().numpy()
        
        """comments: 
            `envs` take a step according to `actions`
        """
        next_obs, rewards, dones, infos = envs.step(actions)
        # envs.render() # close render during training
        
        if dones:
            print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
            writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)
        
        """comments: 
            Add this step's infomation to Replay Buffer.
        """
        rb.add(obs, next_obs, actions, rewards, dones, infos)
        
        """comments: 
            If the game hasn't done, then get the current state.
        """
        obs = next_obs if not dones else envs.reset()
        
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            
            """comments: 
                Sample to get datas from Replay Buffer as a batch of datas.
            """
            data = rb.sample(args.batch_size)
            
            """comments: 
                Calculate $\max Q(s,a)$ using `target_network`, and use `old_val` and `td_target` to calculate 
                Mean-Squared Loss.
            """
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            """comments: 
                Log the current datas into tensorboard, which can be visualized after running.
                We can see that the current directory has a new folder named `runs`, which is used to store the log file.
                https://pytorch.org/docs/stable/tensorboard.html
            """
            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
            
            """comments: 
                This three commands are ofter used together.
                1. Set the gradient to zero.
                2. Backpropogate the loss and get each parameter's gradient.
                3. Update the parameters.
            """
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            """comments: 
                If we have steps with a number of `args.target_network_frequency`, 
                we update target NN with `q_network` 
            """
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())
    if not os.path.exists("./output"):
        os.mkdir("./output")
    torch.save(target_network.state_dict(), './output/dqn.pth')

    """close the env and tensorboard logger"""
    # envs.close()
    writer.close()

    ################## TEST ######################
    obs = envs.reset()
    dones = False 
    envs.render()
    while not dones:
        q_values = target_network(torch.Tensor(obs).to(device))
        actions = torch.argmax(q_values, dim=0).cpu().numpy()
        obs, rewards, dones, infos = envs.step(actions)
        envs.render()
    envs.close()
