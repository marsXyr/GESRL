import os, argparse
import numpy as np
import gym
from gym import wrappers
from utils.filter import Filter
from utils import logz
from utils.tools import get_output_folder


class Noise:
    def __init__(self, shape):
        self.shape = shape

    def sample(self, num):
        return [np.random.randn(*self.shape) for _ in range(num)]


class Policy:
    def __init__(self, state_dim, action_dim, args):
        self.w_policy = np.zeros((action_dim, state_dim))
        self.lr = args.lr

    def __call__(self, state):
        return self.w_policy.dot(state)

    def update(self, pos_rewards, neg_rewards, epsilons, std_rewards):
        grads = np.zeros(self.w_policy.shape)
        for i in range(len(epsilons)):
            grads += (pos_rewards[i] - neg_rewards[i]) * epsilons[i]
        self.w_policy += self.lr / (len(epsilons) * std_rewards) * grads


class ARS:
    def __init__(self):

        self.seed = args.seed
        np.random.seed(self.seed)

        # Init gym env and set the env seed
        self.env = gym.make(args.env)
        self.env.seed(self.seed)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # Init parameters
        self._init_parameters()

        # Init filter, normalizes the input states by tracking the mean and std of states.
        self.filter = Filter(self.state_dim)
        # Init policy, we use linear policy here
        self.policy = Policy(self.state_dim, self.action_dim, args)
        # Init the noise generator
        self.noise = Noise(self.policy.w_policy.shape)

    def _init_parameters(self):

        self.log_dir = args.dir_path
        # The max steps per episode
        self.max_ep_len = args.max_ep_len
        self.epochs = args.epochs
        self.save_freq = args.save_freq
        self.pop_size = args.pop_size
        self.elite_size = args.elite_size
        self.noise_std = args.noise_std

    def evaluate(self):
        state, done, ep_reward, ep_len = self.env.reset(), False, 0.0, 0
        while not done and ep_len < self.max_ep_len:
            self.filter.push(state)
            state = self.filter(state)
            action = self.policy(state)
            state, reward, done, _ = self.env.step(action)
            ep_reward += reward
            ep_len += 1
        return ep_reward, ep_len

    def train(self):

        for epoch in range(self.epochs):
            # Sample noises from the noise generator.
            epsilons = self.noise.sample(self.pop_size)

            pos_rewards, neg_rewards = [], []
            policy_weights = self.policy.w_policy
            # Generate 2 * pop_size policies and rollouts.
            for epsilon in epsilons:
                self.policy.w_policy = policy_weights + self.noise_std * epsilon
                pos_reward, pos_len = self.evaluate()
                pos_rewards.append(pos_reward)

                self.policy.w_policy = policy_weights - self.noise_std * epsilon
                neg_reward, neg_len = self.evaluate()
                neg_rewards.append(neg_reward)
            self.policy.w_policy = policy_weights

            std_rewards = np.array(pos_rewards + neg_rewards).std()

            # ARS update
            if self.elite_size != 0:
                scores = {k: max(pos_reward, neg_reward) for k, (pos_reward, neg_reward) in enumerate(zip(pos_rewards, neg_rewards))}
                sorted_scores = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:self.elite_size]
                elite_pos_rewards = [pos_rewards[k] for k in sorted_scores]
                elite_neg_rewards = [neg_rewards[k] for k in sorted_scores]
                elite_epsilons = [epsilons[k] for k in sorted_scores]
                self.policy.update(elite_pos_rewards, elite_neg_rewards, elite_epsilons, std_rewards)
            else:

                self.policy.update(pos_rewards, neg_rewards, epsilons, std_rewards)

            # Save policy and log the information
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
    parser.add_argument('--noise_std', type=float, default=0.03)
    parser.add_argument('--pop_size', type=int, default=16)
    parser.add_argument('--elite_size', type=int, default=16)
    parser.add_argument('--max_ep_len', type=int, default=1000)

    # Experiment setting
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dir_path', type=str, default='results/')

    args = parser.parse_args()

    output_path = args.dir_path
    for seed in range(1, 11):
        args.seed = seed
        args.dir_path = get_output_folder(output_path, args.env, args.seed)
        logz.configure_output_dir(args.dir_path)
        logz.save_params(vars(args))
        ars = ARS()
        ars.train()
