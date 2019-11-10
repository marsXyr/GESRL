import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F


#USE_CUDA = torch.cuda.is_available()
USE_CUDA = False
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


def to_numpy(var):
    return var.cpu().data.numpy()


def to_tensor(x):
    return torch.FloatTensor(x)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class RLNN(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(RLNN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

    def set_params(self, w):
        for i, param in enumerate(self.parameters()):
            param.data.copy_(torch.from_numpy(w).view(param.size()))

    def get_params(self):
        params = [to_numpy(v) for v in self.parameters()]
        return deepcopy(params[0])

    def get_grads(self):
        grads = [to_numpy(v.grad) for v in self.parameters()]
        return deepcopy(grads[0])

    def get_size(self):
        return self.get_params().shape[0]

    def load_model(self, filename, net_name):
        if filename is None:
            return

        self.load_state_dict(
            torch.load('{}/{}.pkl'.format(filename, net_name), map_location=lambda storage, loc: storage))

    def save_model(self, output, net_name):
        torch.save(self.state_dict(), '{}/{}.pkl'.format(output, net_name))


class LinearPolicy(RLNN):
    """
    Linear policy class that computes action as <w, ob>.
    """

    def __init__(self, state_dim, action_dim, max_action, args):
        super(LinearPolicy, self).__init__(state_dim, action_dim)

        self.l1 = nn.Linear(self.state_dim, self.action_dim, bias=False)

        self.optimizer = Adam(self.parameters(), lr=args.actor_lr)
        self.tau = args.tau
        # self.theta = args['theta']
        self.max_action = max_action
        if USE_CUDA:
            self.cuda()

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

    def update(self, memory, batch_size, critic, policy_t):
        # Sample replay buffer
        states, _, _, _, _ = memory.sample(batch_size)

        # Compute actor loss
        policy_loss = -critic(states, self(states)).mean()

        # Optimize the policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        grads = self.get_grads()   # Get policy gradients
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), policy_t.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return grads


class Critic(RLNN):

    def __init__(self, state_dim, action_dim, args):
        super(Critic, self).__init__(state_dim, action_dim)
        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(64)
            self.n2 = nn.LayerNorm(64)
        self.layer_norm = args.layer_norm
        self.optimizer = Adam(self.parameters(), lr=args.critic_lr)
        self.tau = args.tau
        self.discount = args.discount
        if USE_CUDA:
            self.cuda()

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

    def update(self, memory, batch_size, policy, critic_t):
        # Sample replay buffer
        states, n_states, actions, rewards, dones = memory.sample(batch_size)

        # Q target = reward + discount * Q(next_state, pi(next_state))
        with torch.no_grad():
            target_Q = critic_t(n_states, policy(n_states))
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        # Get current Q estimate
        current_Q = self.forward(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        # Optimize the critic
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), critic_t.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
