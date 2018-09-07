'''
Written by chongyou

Q-Learning Agent with epsilon-greedy policy to solve planeworld environment.

Learns by moving each plane individually while taking other planes as obstacles
and iterating learning process over all planes.
All planes move simultaneously in a time-step



'''
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from itertools import count

import planeworld

plt.ion()

class Agent:
    def __init__(self, epsilon = 0.05, alpha = 0.5, gamma = 1):
        self.actions = env.action_space
        self.num_actions = len(self.actions)
        self.Q = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def _stateInt(self, state):  # takes in numpy array state
        state.tolist()           # and converts to unique integer
        output = ''
        for row in range(len(state)):
            for col in range(len(state[row])):
                cell = int(state[row][col])
                output += str(cell)
        return int(output, base = 3)

    def act(self, state): # takes in numpy array state
        # epsilon greedy policy
        stateStr = self._stateInt(state)
        if np.random.binomial(1,self.epsilon != 1):
            for acts in self.actions:
                if (stateStr, acts) not in self.Q:
                    self.Q[(stateStr,acts)] = 0.0 # initialise in Q
            action_dict = {a:v for ((s,a),v) in self.Q.items() if s == stateStr}
            max_v = action_dict[max(action_dict, key = action_dict.get)]
            action_max_list = [a for (a,v) in action_dict.items() if v == max_v]
            action = np.random.choice(action_max_list)
        else:
            action = np.random.choice(self.actions)
            #print('eps')
        #print('act:'+str(action))
        return action

    def learn(self, state1, action1, reward, state2):
        state1 = self._stateInt(state1)
        state2 = self._stateInt(state2)

        for acts in self.actions: # initialise in Q
            if (state2, acts) not in self.Q:
                self.Q[(state2,acts)] = 0.0
        if (state1,action1) not in self.Q:
            self.Q[(state1,action1)] = 0.0

        qMax = max([v for ((s,a),v) in self.Q.items() if s == state2])
        td_delta = reward + self.gamma * qMax - self.Q[(state1,action1)]
        self.Q[(state1,action1)] += self.alpha * td_delta

    def episode(self, max_steps): # not episodic but rather continuous

        env.render()

        rewards_dict = {}
        current_plane_no = 0
        
        for t in count():
            print(t)

            plane_dict = env.get_planes().copy()

            if current_plane_no != env.no_planes: # for graphing
                for plane in plane_dict:
                    if plane not in rewards_dict:
                        rewards_dict[plane] = 0
                current_plane_no = env.no_planes

            action_dict = {}
            state1_dict = {}

            # get state1 and actions
            for plane in plane_dict:
                state1 = env.state_specific(plane)
                state1_dict[plane] = state1
                action1 = self.act(state1)
                action_dict[plane] = action1

            # execute actions
            output = env.step(action_dict)

            # get state2 and learn
            for plane in plane_dict:
                state2 = env.state_specific(plane)
                reward = output[plane]['reward']
                self.learn(state1_dict[plane],action_dict[plane],reward,state2)
                rewards_dict[plane] += reward
            #env.render()
            #input('Pause')

            if t >= max_steps:
                with open('agent_output.txt', 'w') as f:
                    f.write(str(self.Q))
                break

        print('goals='+str(env.no_goals)+'.crash='+str(env.no_crashes)+'.total='+str(env.no_planes))

        rew = []
        rew_no = []

        for plane_no in range(0, env.no_planes+1, 10):
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

if __name__ == '__main__':
    env = planeworld.PlaneEnv()
    A = Agent()
    A.episode(50000)

