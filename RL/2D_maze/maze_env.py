#!/usr/bin/env python3

import numpy as np
import time
import sys
import tkinter as tk


UNIT = 40
MAZE_HEIGHT = 8
MAZE_WIDTH = 8


class Maze(tk.Tk):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_HEIGHT * UNIT, MAZE_HEIGHT * UNIT))
        self._build_maze()

    def _create(self, center, fill, name, mode='rectangle'):
        if mode == 'rectangle':
            rec = self.canvas.create_rectangle(
                center[0] - 15, center[1] - 15,
                center[0] + 15, center[1] + 15, 
                fill = fill
            )
            self.__setattr__(name, rec)
        elif mode == 'oval':
            oval = self.canvas.create_oval(
                center[0] - 15, center[1] - 15,
                center[0] + 15, center[1] + 15, 
                fill = fill
            )
            self.__setattr__(name, oval)

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_HEIGHT * UNIT, width=MAZE_WIDTH * UNIT)

        for c in range(0, MAZE_WIDTH * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_HEIGHT * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        
        for r in range(0, MAZE_HEIGHT * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_WIDTH * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        origin = np.array([20, 20])

        self._create(origin + np.array([UNIT * 2, UNIT]), 'black', 'hell1')
        self._create(origin + np.array([UNIT, UNIT * 2]), 'black', 'hell2')
        self._create(origin + np.array([UNIT, UNIT * 6]), 'black', 'hell4')
        self._create(origin + np.array([UNIT * 4, UNIT * 2]), 'black', 'hell5')
        self._create(origin + UNIT * 2, 'yellow', 'oval', mode='oval')
        self._create(origin, 'red', 'rect')
        self.canvas.pack()

    def reset(self, episode):
        self.title('%s episode-%d' % ('maze', episode))
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self._create(origin, 'red', 'rect')

       # return self.canvas.coords(self.rect)
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_HEIGHT*UNIT)

    def step(self, action):
        state = self.canvas.coords(self.rect)
        base_actions = np.array([0, 0])

        cross_border = False
        if action == 0:
            # up
            if state[1] > UNIT:
                base_actions[1] -= UNIT
            else:
                cross_border = True
        elif action == 1:
            # down
            if state[1] < (MAZE_HEIGHT - 1) * UNIT:
                base_actions[1] += UNIT
            else:
                cross_border = True
        elif action == 2:
            # right
            if state[0] < (MAZE_WIDTH - 1) * UNIT:
                base_actions[0] += UNIT
            else:
                cross_border = True
        elif action == 3:
            # left
            if state[0] > UNIT:
                base_actions[0] -= UNIT
            else:
                cross_border = True
        '''
        if cross_border:
            reward = -1
            done = True
            state_ = 'terminal'
            return state_, reward, done
        '''

        self.canvas.move(self.rect, base_actions[0], base_actions[1])
        
        state_ = self.canvas.coords(self.rect)

        if state_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            state_ = 'terminal'
        elif state_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2), 
            self.canvas.coords(self.hell4), self.canvas.coords(self.hell5)]:
            reward = -1
            done = True
            state_ = 'terminal'
        else:
            reward = 0
            done = False
        
        next_coords = self.canvas.coords(self.rect)
        state_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_HEIGHT*UNIT)
        return state_, reward, done
    
    def render(self):
        time.sleep(0.1)
        self.update()