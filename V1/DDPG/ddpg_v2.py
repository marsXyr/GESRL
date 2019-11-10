import argparse
import os, time, random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.tools import hard_update, soft_update, OUNoise, get_output_folder
from utils.buffer import ReplayBuffer
from utils import logz

FloatTensor = torch.FloatTensor


"""
    Actor 和 Critic 一起训练
"""


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action, args):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, action_dim, bias=False)
        self.max_action = max_action

    def forward(self, x):

        out = self.l1(x)
        # abs_out = torch.abs(out)
        # abs_out_sum = torch.sum(abs_out).view(-1, 1)
        # abs_out_mean = abs_out_sum / self.action_dim / self.theta
        # ones = torch.ones(abs_out_mean.size())
        # ones = ones.cuda()
        # mod = torch.where(abs_out_mean >= 1, abs_out_mean, ones)
        # out = out / mod
        #
        out = self.max_action * torch.tanh(out)
        return out

    def get_grads(self):
        grads = [v.grad.data.numpy() for v in self.parameters()]
        return grads[0]

    def get_params(self):
        params = [v.data.numpy() for v in self.parameters()]
        return params[0]

    def set_params(self, w):
        for i, param in enumerate(self.parameters()):
            param.data.copy_(torch.from_numpy(w).view(param.size()))


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, args):
        super(Critic, self).__init__()
        l1_dim, l2_dim = 400, 300
        self.l1 = nn.Linear(state_dim + action_dim, l1_dim)
        self.l2 = nn.Linear(l1_dim, l2_dim)
        self.l3 = nn.Linear(l2_dim, 1)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(l1_dim)
            self.n2 = nn.LayerNorm(l2_dim)
        self.layer_norm = args.layer_norm

    def forward(self, x, u):
        if not self.layer_norm:
            x = F.leaky_relu(self.l1(torch.cat([x, u], 1)))
            x = F.leaky_relu(self.l2(x))
            x = self.l3(x)
        else:
            x = F.leaky_relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x = F.leaky_relu(self.n2(self.l2(x)))
            x = self.l3(x)
        return x


class DDPG:

    def __init__(self, state_dim, action_dim, max_action, args):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self._init_parameters(args)
        self._init_nets(args)

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.state_dim, self.action_dim)

    def _init_parameters(self, args):
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.discount = args.discount
        self.tau = args.tau
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size

    def _init_nets(self, args):
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action, args)
        self.actor_t = Actor(self.state_dim, self.action_dim, self.max_action, args)

        self.critic = Critic(self.state_dim, self.action_dim, args)
        self.critic_t = Critic(self.state_dim, self.action_dim, args)

        hard_update(self.actor_t, self.actor)
        hard_update(self.critic_t, self.critic)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.loss = nn.MSELoss()

    def train(self):
        states, n_states, actions, rewards, dones = self.replay_buffer.sample(self.batch_size)
        # Compute q target
        ##########################
        next_q = self.critic_t(n_states, self.actor_t(n_states))
        q_target = (rewards + self.discount * (1 - dones.float()) * next_q).detach()
        # Compute q predict
        q_predict = self.critic(states, actions)

        # Critic update
        critic_loss = self.loss(q_predict, q_target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor update
        actor_loss = - self.critic(states, self.actor(states)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        actor_grad = self.actor.get_grads()
        self.actor_optim.step()

        return actor_grad

    def update_nets(self):
        soft_update(self.actor_t, self.actor, self.tau)
        soft_update(self.critic_t, self.critic, self.tau)



def run(args):

    log_dir = args.dir_path

    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])

    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ddpg = DDPG(state_dim, action_dim, max_action, args)
    ounoise = OUNoise(action_dim)

    def get_action(state, noise=None):
        action = ddpg.actor(FloatTensor(state))
        action = (action.data.numpy() + noise.add()) if noise else action.data.numpy()
        return np.clip(action, -max_action, max_action)

    def rollout(eval=False):
        state, done, ep_reward, ep_len = env.reset(), False, 0.0, 0
        while not done and ep_len < args.max_ep_len:
            if not eval:
                action = get_action(state, noise=ounoise)
            else:
                action = get_action(state)
            next_state, reward, done, _ = env.step(action)
            if not eval:
                done = False if ep_len + 1 == args.max_ep_len else done
                ddpg.replay_buffer.store((state, next_state, action, reward, done))
            ep_reward += reward
            ep_len += 1
            state = next_state
        return ep_reward, ep_len

    for epoch in range(args.epochs):
        ep_reward, ep_len = rollout(eval=False)
        if epoch > args.start_epoch:
            for _ in range(ep_len):
                ddpg.train()
                ddpg.update_nets()

        if epoch % args.save_freq == 0:
            test_rewards = []
            for i in range(10):
                reward, _ = rollout()
                test_rewards.append(reward)
            test_rewards = np.array(test_rewards)

            np.savez(log_dir + '/policy_weights', ddpg.actor.get_params())
            logz.log_tabular("Epoch", epoch)
            logz.log_tabular("AverageTestReward", np.mean(test_rewards))
            logz.log_tabular("StdTestRewards", np.std(test_rewards))
            logz.log_tabular("MaxTestRewardRollout", np.max(test_rewards))
            logz.log_tabular("MinTestRewardRollout", np.min(test_rewards))
            logz.dump_tabular()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')

    parser.add_argument('--actor_lr', type=float, default=0.0001)
    parser.add_argument('--critic_lr', type=float, default=0.0001)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--layer_norm', type=bool, default=True)

    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--start_epoch', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dir_path', type=str, default='results_v2/')

    args = parser.parse_args()

    output_path = args.dir_path
    for seed in range(1, 11):
        args.seed = seed
        args.dir_path = get_output_folder(output_path, args.env, args.seed)
        logz.configure_output_dir(args.dir_path)
        logz.save_params(vars(args))
        run(args)


