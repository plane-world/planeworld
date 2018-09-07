'''
Written by chongyou

Deep Q-Learning Agent with epsilon-greedy policy (annealed epsilon)
to solve planeworld environment.

Written with tensorflow and tf.keras

Learns by moving each plane individually while taking other planes as obstacles
and iterating learning process over all planes.
All planes move simultaneously in a time-step.

'''

import planeworld

import tensorflow as tf
from tensorflow import keras

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class DQNAgent:

    def _state_rep(self, state): # convert state rep for tf
        return np.reshape(state, (1, self.nS))

    def __init__(self, env,  epsilon=0.5, alpha=0.1, gamma=0.95):

        self.actions = env.action_space
        self.nA = env.nA
        self.nS = env.nS

        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.alpha = alpha
        self.gamma = gamma

        # replay memory and learning
        self.memory = np.zeros((0, 5))
        self.batch_size = 64
        self.start_size = 500
        self.max_size = 5000
        self.update_target_rate = 50

        # initialise model
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(self.nS, input_dim=self.nS, activation='tanh'))
        self.model.add(keras.layers.Dense(self.nS, activation='tanh'))
        self.model.add(keras.layers.Dense(self.nA, activation='softmax'))
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    def act(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def learn(self):
        minibatch = self.memory[np.random.choice(self.memory.shape[0], self.batch_size), :]
        for state1, action, reward, state2, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(state2)[0])
            target_f = self.model.predict(state1)
            target_f[0][action] = target
            self.model.fit(state1, target_f, epochs=1, verbose=0)

    def run(self, max_steps):
        rewards_dict = {}
        current_plane_no = 0

        for step in range(max_steps):

            # anneal epsilon
            if step >= self.start_size:
                epsilon = self.epsilon * (self.start_size / step)
                epsilon = max(epsilon, self.epsilon_min)
            else:
                epsilon = self.epsilon

            # initialise dicts
            plane_dict = env.get_planes().copy()
            action_dict = {}
            state1_dict = {}

            # reward graphing initialisation
            if current_plane_no != env.no_planes:
                for plane in plane_dict:
                    if plane not in rewards_dict:
                        rewards_dict[plane] = 0
                current_plane_no = env.no_planes

            # get state1 and actions
            for plane in plane_dict:
                state1 = self._state_rep(env.state_specific(plane))
                state1_dict[plane] = state1

                if np.random.rand() < epsilon:  # epsilon greedy
                    action1 = np.random.choice(self.actions)
                else:
                    action1 = self.act(state1)

                action_dict[plane] = action1

            # execute actions
            output = env.step(action_dict)

            # get state2 and learn
            for plane in plane_dict:
                state2 = self._state_rep(env.state_specific(plane))

                # variables
                action = action_dict[plane]
                reward = output[plane]['reward']
                done = output[plane]['done']

                # add reward to rewards dict
                rewards_dict[plane] += reward

                # trimming replay memory, probably will cause overfitting
                if len(self.memory) >= self.max_size:
                    np.delete(self.memory, np.random.randint(self.max_size / 2), 0)

                # add to experience replay
                D = (state1, action, reward, state2, done)
                self.memory = np.append(self.memory, [D], axis=0)

            if step >= self.start_size and step % 50 == 0:
                print(step)
                self.learn()

        # save to file
        self.model.save_weights('./weights')

        # print
        print('goals=' + str(env.no_goals) + '.crash=' + str(env.no_crashes) + '.total=' + str(env.no_planes))

        rew = []
        rew_no = []

        for plane_no in range(0, env.no_planes + 1, 10):
            rew_no.append(plane_no)
            if plane_no in rewards_dict:
                rew.append(rewards_dict[plane_no])
            else:
                rew.append(0)

        plt.figure(1)
        plt.title('episode vs rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(rew_no, rew, 'bx')
        plt.waitforbuttonpress()
        plt.clf()


if __name__ == "__main__":
    plt.ion()
    env = planeworld.PlaneEnv()
    agent = DQNAgent(env)
    agent.run(50000)





