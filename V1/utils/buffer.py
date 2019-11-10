import numpy as np
import torch


FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor

# Code based on https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
# and https://github.com/jingweiz/pytorch-distributed/blob/master/core/memories/shared_memory.py


class ReplayBuffer:

    def __init__(self, memory_size, state_dim, action_dim):

        # params
        self.memory_size = memory_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pos = 0
        self.full = False

        self.states = torch.zeros(self.memory_size, self.state_dim)
        self.actions = torch.zeros(self.memory_size, self.action_dim)
        self.n_states = torch.zeros(self.memory_size, self.state_dim)
        self.rewards = torch.zeros(self.memory_size, 1)
        self.dones = torch.zeros(self.memory_size, 1)

    def store(self, datum):

        state, n_state, action, reward, done = datum

        self.states[self.pos] = FloatTensor(state)
        self.n_states[self.pos] = FloatTensor(n_state)
        self.actions[self.pos] = FloatTensor(action)
        self.rewards[self.pos] = FloatTensor([reward])
        self.dones[self.pos] = FloatTensor([done])

        self.pos += 1
        if self.pos == self.memory_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):

        upper_bound = self.memory_size if self.full else self.pos
        batch_inds = LongTensor(np.random.randint(0, upper_bound, size=batch_size))

        return (self.states[batch_inds],
                self.n_states[batch_inds],
                self.actions[batch_inds],
                self.rewards[batch_inds],
                self.dones[batch_inds])

    def buffer_flush(self):
        self.pos = 0
        self.full = False

        self.states = torch.zeros(self.memory_size, self.state_dim)
        self.actions = torch.zeros(self.memory_size, self.action_dim)
        self.n_states = torch.zeros(self.memory_size, self.state_dim)
        self.rewards = torch.zeros(self.memory_size, 1)
        self.dones = torch.zeros(self.memory_size, 1)


