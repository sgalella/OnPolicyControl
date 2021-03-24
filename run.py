import gym
import numpy as np
from tqdm import tqdm

from tiles import IHT, tiles


class Tiles:
    def __init__(self, size, num_tiles, num_tilings):
        self.iht = IHT(size)
        self.num_tiles = num_tiles
        self.num_tilings = num_tilings

    def get_features(self, pos, vel):
        return np.array(tiles(self.iht, self.num_tiles, [self.num_tilings * pos / (0.5 + 1.2), self.num_tilings * vel / (0.07 + 0.07)]))


class SemiGradientSarsa:
    def __init__(self, alpha, epsilon, gamma, num_actions, size):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_actions = num_actions
        self.w = np.zeros((num_actions, size))
        self.actions = np.array(range(num_actions))
        self.last_q = None
        self.last_x = None
        self.current_q = None
        self.current_x = None
        self.last_action = None

    def initialize_episode(self, indices, action):
        self.last_action = action
        self.last_x = np.zeros_like(self.w)
        self.last_x[:, indices] = 1
        self.last_q = np.sum(np.multiply(self.w, self.last_x), axis=1)

    def get_action(self, indices):
        self.current_x = np.zeros_like(self.w)
        self.current_x[:, indices] = 1
        self.current_q = np.sum(np.multiply(self.w, self.current_x), axis=1)
        if np.random.random() > self.epsilon:
            return np.argmax(self.current_q)
        return np.random.randint(self.num_actions)

    def update_step(self, new_action, reward):
        self.w[self.last_action] += self.alpha * (reward + self.gamma * self.current_q[new_action] - self.last_q[self.last_action]) * self.last_x[self.last_action]
        self.last_q = self.current_q
        self.last_x = self.current_x
        self.last_action = new_action

    def update_end(self, reward):
        self.w[self.last_action] += self.alpha * (reward - self.last_q[self.last_action]) * self.last_x[self.last_action]


def main(env, tiles, agent):
    num_episodes = 500
    num_steps = 200

    for episode in tqdm(range(num_episodes)):
        # Initial state and action of episode
        observation = env.reset()
        action = env.action_space.sample()
        pos, vel = observation
        indices = tiles.get_features(pos, vel)
        agent.initialize_episode(indices, action)

        for step in range(num_steps):
            if episode == num_episodes - 1:
                env.render()

            # Take action A, observe R and S'
            new_observation, reward, done, info = env.step(action)
            new_pos, new_vel = new_observation
            new_indices = tiles.get_features(new_pos, new_vel)

            if done is True:
                agent.update_end(reward)
                break

            # Choose A' as a function of q
            new_action = agent.get_action(new_indices)
            agent.update_step(new_action, reward)
            action = new_action

    env.close()


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    tls = Tiles(size=4096, num_tiles=8, num_tilings=8)
    agent = SemiGradientSarsa(alpha=0.5 / 8, epsilon=0.1, gamma=1, num_actions=env.action_space.n, size=4096)
    main(env, tls, agent)
