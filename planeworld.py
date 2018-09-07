import numpy as np
import sys
import random

LENGTH = 5
MAX_PLANES = 3
SPAWN_CHANCE = 2 # reciprocal of spawn chance

class Plane:

    def __init__(self, length=LENGTH):
        # initialise plane coords from edge of environment
        self.length = length
        self.coords = random.choice([(x, y) for x in range(self.length - 1) for y in range(self.length - 1) if
                                     (x == 0 or x == self.length - 1 or y == 0 or y == self.length - 1)])

    def get_coords(self): # get coordinates of plane
        return self.coords

    def move_to(self, new_coords):  # moves planes to coords
        self.coords = new_coords


class PlaneEnv:

    # metadata = {'render.modes': ['human', 'ansi', 'rgb_array']}
    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.length - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.length - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def __init__(self):
        self.length = LENGTH
        self.shape = (self.length, self.length)
        self.goal = ((self.length - 1) / 2, (self.length - 1) / 2)

        self.nS = np.prod(self.shape)
        self.nA = 4
        self.action_space = list(range(self.nA))

        self.reset()

    def get_planes(self): # returns dictionary of planes
        return self.plane_dict

    def add_plane(self): # adds a plane
        if len(self.plane_dict) < MAX_PLANES:
            plane_no = 1
            while plane_no in self.plane_dict:
                plane_no += 1
            coord_list = list(map(lambda x: x[1].coords, self.plane_dict.items()))
            # don't spawn a plane on another plane
            new_plane = Plane()
            while new_plane.coords in coord_list:
                new_plane = Plane()
            self.plane_dict[plane_no] = new_plane

    def move_plane(self, action, plane_name): # moves plane
        delta = {0: [-1, 0],    # up
                 1: [0, 1],     # right
                 2: [1, 0],     # down
                 3: [0, -1]     # left
                 }.get(action, [0, 0])
        plane = self.plane_dict[plane_name]
        new_position = np.array(plane.coords) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        plane.move_to(tuple(new_position))

    def reset(self): # resets state to beginning
        self.plane_dict = {}
        self.add_plane()
        return self.state_general()

    def step(self, actions_dict, spawn_new=True):  # actions is dict of format plane:action

        current_planes = list(map(lambda x: x[0], self.plane_dict.items()))

        # execute actions 'concurrently'
        for plane in current_planes:
            self.move_plane(actions_dict[plane], plane)

        # initalise output dictionary
        output = dict(map(lambda x: (x[0], {'reward': -1, 'done': False}), self.plane_dict.items()))
        # create dictionary of format {coordinate : [planes in coordinate]} to check for coordinate collision
        coords_dict = dict(map(lambda x: (x, list(filter(lambda y: y[1].coords == x, self.plane_dict.items()))),
                               sorted(set(map(lambda z: z[1].coords, self.plane_dict.items())))))

        # assign rewards and done
        for plane in current_planes:
            coords = self.plane_dict[plane].coords
            if coords == self.goal:  # goal check
                output[plane]['reward'] = +10
                output[plane]['done'] = True
                self.plane_dict.pop(plane)
            elif len(coords_dict[coords]) > 1:  # collision check
                output[plane]['reward'] = -10   # Planes can swap positions without penalty
                output[plane]['done'] = True    # but just cannot be in the same square.
                self.plane_dict.pop(plane)
            else:
                output[plane]['reward'] = -1  # continue

        # don't spawn new planes if spawn_new == False
        if spawn_new and (np.random.randint(0, SPAWN_CHANCE) == 0 or not self.plane_dict):
            self.add_plane()

        return output  # output: {plane:{reward:_, done:_}}

    def state_specific(self, plane):    # returns state specific to plane
        state = np.zeros(self.shape)
        for plane_iter in self.plane_dict:
            if plane_iter == plane:
                state[self.plane_dict[plane_iter].coords] = 2  # plane
            else:
                state[self.plane_dict[plane_iter].coords] = 1  # obstacles
        return state

    def state_general(self): # general state
        state = np.zeros(self.shape)
        for plane_iter in self.plane_dict:
            state[self.plane_dict[plane_iter].coords] = plane_iter
        return state

    def render(self, plane=None):   # prints out environment onto stdout

        outfile = sys.stdout

        if plane in self.plane_dict:    # prints specific state
            state = self.state_specific(plane)
            state.tolist()
            for row in range(len(state)):
                for col in range(len(state[row])):
                    cell = state[row][col]
                    if cell == 1:
                        output = ' x '
                    elif cell == 2:
                        output = ' P '
                    elif (row, col) == self.goal:
                        output = ' T '
                    else:
                        output = ' . '

                    if col == 0:
                        output = output.lstrip()
                    elif col == self.length - 1:
                        output = output.rstrip()
                        output += '\n'

                    outfile.write(output)
        else:   # prints general state
            state = self.state_general()
            state.tolist()
            for row in range(len(state)):
                for col in range(len(state[row])):
                    cell = int(state[row][col])
                    if cell != 0:
                        output = ' ' + str(cell) + ' '
                    elif (row, col) == self.goal:
                        output = ' T '
                    else:
                        output = ' . '

                    if col == 0:
                        output = output.lstrip()
                    elif col == self.length - 1:
                        output = output.rstrip()
                        output += '\n'

                    outfile.write(output)

        outfile.write('\n')