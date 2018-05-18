#!/usr/bin/env python3

import numpy as np
import pandas as pd
import time


np.random.seed(2)


N_STATES = 6
ACTIONS = ['left', 'right']
EPSILON = 0.9   # greedy policy
ALPHA = 0.1     #learning rate
LAMBDA = 0.9
MAX_EPISODES = 13
FRESH_TIME = 0.3

def build_q_table(n_states, actions):
    tables = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions
    )
    return tables


def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name


def get_env_feedback(state, action):
    if action == 'right':
        if state == N_STATES -2:
            state = 'terminal'
            reward = 1
        else:
            state = state + 1
            reward = 0
    else:
        reward = 0
        if state == 0:
            state = state
        else:
            state = state - 1
    return state, reward

def update_env(state, episode, step_counter):
    env_list = ['-'] * (N_STATES - 1) + ['T']
    if state == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        state = 0
        is_terminated = False
        update_env(state, episode, step_counter)
        while not is_terminated:
            action = choose_action(state, q_table)
            state_, reward = get_env_feedback(state, action)
            q_predict = q_table.ix[state, action]
            if state_ != 'terminal':
                q_target = reward + LAMBDA * q_table.iloc[state_, :].max()
            else:
                q_target = reward
                is_terminated = True
            
            q_table.ix[state, action] += ALPHA * (q_target - q_predict)
            state = state_
            
            step_counter += 1
            update_env(state, episode, step_counter)

    return q_table


if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ_table:\n')
    print(q_table)