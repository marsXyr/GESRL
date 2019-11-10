import argparse
import numpy as np
import gym
from DDPG.ddpg_v1 import DDPG
from utils import logz
from utils.filter import Filter
from utils.tools import get_output_folder

"""
    ddpg_v1 + 用ddpg梯度作为代理梯度 + 用RL更新后的actor替换现有policy + 无 buffer flush
"""

class Noise:
    def __init__(self, size, args):
        self.size = size
        self.alpha = args.alpha
        self.sigma = args.sigma
        self.k = args.k

        self.U = np.zeros((self.size, self.k))

    def sample(self, num):
        noises = np.sqrt(self.alpha / self.size) * np.random.randn(num, self.size) + \
                 np.sqrt((1 - self.alpha) / self.k) * np.random.randn(num, self.k) @ self.U.T

        return noises * self.sigma

    def update(self, grads):
        self.U, _ = np.linalg.qr(np.array(grads).reshape(self.size, self.k))


class Policy:
    def __init__(self, state_dim, action_dim, args):
        self.w_policy = np.zeros((action_dim, state_dim))
        self.lr = args.lr
        self.beta = args.beta
        self.sigma = args.sigma

    def __call__(self, state):
        return self.w_policy.dot(state)

    def update_by_ges(self, pos_rewards, neg_rewards, epsilons, std_rewards):
        grads = np.zeros(self.w_policy.shape)
        for i in range(len(epsilons)):
            grads += (pos_rewards[i] - neg_rewards[i]) * epsilons[i].reshape(self.w_policy.shape)
        self.w_policy += self.lr * (self.beta / (2 * self.sigma ** 2 * len(epsilons) * std_rewards)) * grads

    def update_by_ddpg(self, grads):
        self.w_policy += grads


class GESRL:
    def __init__(self):

        self.seed = args.seed
        np.random.seed(self.seed)

        # Init gym env and set the env seed
        self.env = gym.make(args.env)
        self.env.seed(self.seed)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = int(self.env.action_space.high[0])

        # Init parameters
        self._init_parameters()

        self.filter = Filter(self.state_dim)
        self.policy = Policy(self.state_dim, self.action_dim, args)
        self.noise = Noise(self.policy.w_policy.size, args)
        self.ddpg = DDPG(self.state_dim, self.action_dim, self.max_action, args)

    def _init_parameters(self):

        self.log_dir = args.dir_path
        # The max steps per episode
        self.max_ep_len = args.max_ep_len
        self.epochs = args.epochs
        self.save_freq = args.save_freq
        self.start_epoch = args.start_epoch
        self.rl_train_steps = args.rl_train_steps
        self.pop_size = args.pop_size
        self.elite_size = args.elite_size
        # subspace dimension
        self.k = args.k

    def evaluate(self, eval=False):
        state, done, ep_reward, ep_len = self.env.reset(), False, 0.0, 0
        while not done and ep_len < self.max_ep_len:
            self.filter.push(state)
            state = self.filter(state)
            action = self.policy(state)
            next_state, reward, done, _ = self.env.step(action)
            if not eval:
                done = False if ep_len + 1 == args.max_ep_len else done
                self.ddpg.replay_buffer.store((state, next_state, action, reward, done))
            ep_reward += reward
            ep_len += 1
            state = next_state
        return ep_reward, ep_len

    def train(self):

        for epoch in range(self.epochs):
            surr_grads = []
            if epoch >= self.start_epoch:
                self.ddpg.actor.set_params(self.policy.w_policy)
                self.ddpg.actor_t.set_params(self.policy.w_policy)

                for step in range(self.rl_train_steps):
                    grad = self.ddpg.train()
                    if step >= self.rl_train_steps - self.k:
                        surr_grads.append(grad.flatten())

                self.policy.w_policy = self.ddpg.actor.get_params()
                self.noise.update(np.array(surr_grads).T)

            epsilons = self.noise.sample(self.pop_size)  # policy_size x pop_size

            pos_rewards, neg_rewards = [], []
            policy_weights = self.policy.w_policy   # action_dim x state_dim
            for epsilon in epsilons:
                self.policy.w_policy = policy_weights + epsilon.reshape(self.policy.w_policy.shape)
                pos_reward, pos_len = self.evaluate()
                pos_rewards.append(pos_reward)

                self.policy.w_policy = policy_weights - epsilon.reshape(self.policy.w_policy.shape)
                neg_reward, neg_len = self.evaluate()
                neg_rewards.append(neg_reward)
            self.policy.w_policy = policy_weights

            std_rewards = np.array(pos_rewards + neg_rewards).std()

            if self.elite_size != 0:
                scores = {k: max(pos_reward, neg_reward) for k, (pos_reward, neg_reward) in
                          enumerate(zip(pos_rewards, neg_rewards))}
                sorted_scores = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:self.elite_size]
                elite_pos_rewards = [pos_rewards[k] for k in sorted_scores]
                elite_neg_rewards = [neg_rewards[k] for k in sorted_scores]
                elite_epsilons = [epsilons[k] for k in sorted_scores]
                self.policy.update_by_ges(elite_pos_rewards, elite_neg_rewards, elite_epsilons, std_rewards)
            else:
                self.policy.update_by_ges(pos_rewards, neg_rewards, epsilons, std_rewards)

            if epoch % self.save_freq == 0:
                train_rewards = np.array(pos_rewards + neg_rewards)
                test_rewards = []
                for _ in range(10):
                    reward, _ = self.evaluate()
                    test_rewards.append(reward)
                test_rewards = np.array(test_rewards)

                np.savez(self.log_dir + '/policy_weights', self.policy.w_policy)
                logz.log_tabular("Epoch", epoch)
                logz.log_tabular("AverageTrainReward", np.mean(train_rewards))
                logz.log_tabular("StdTrainRewards", np.std(train_rewards))
                logz.log_tabular("MaxTrainRewardRollout", np.max(train_rewards))
                logz.log_tabular("MinTrainRewardRollout", np.min(train_rewards))
                logz.log_tabular("AverageTestReward", np.mean(test_rewards))
                logz.log_tabular("StdTestRewards", np.std(test_rewards))
                logz.log_tabular("MaxTestRewardRollout", np.max(test_rewards))
                logz.log_tabular("MinTestRewardRollout", np.min(test_rewards))
                logz.dump_tabular()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')

    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--actor_lr', type=float, default=0.0001)
    parser.add_argument('--critic_lr', type=float, default=0.0001)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--layer_norm', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--pop_size', type=int, default=16)
    parser.add_argument('--elite_size', type=int, default=16)
    parser.add_argument('--max_ep_len', type=int, default=1000)

    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=2.)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--k', type=float, default=50)

    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--rl_train_steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dir_path', type=str, default='results_v1b/')

    args = parser.parse_args()

    output_path = args.dir_path
    for seed in range(1, 11):
        args.seed = seed
        args.dir_path = get_output_folder(output_path, args.env, args.seed)
        logz.configure_output_dir(args.dir_path)
        logz.save_params(vars(args))
        gesrl = GESRL()
        gesrl.train()

