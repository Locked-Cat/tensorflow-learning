#!/usr/bin/env python3
from maze_env import Maze
from rl import QLearningTable
from rl import SarsaTable
from rl import SarsaLambdaTable
from rl import DeepQNetwork


def update(env: Maze):
    q_table = QLearningTable(actions=list(range(env.n_actions)))
    #q_table = SarsaTable(actions=list(range(env.n_actions)))
    #q_table = SarsaLambdaTable(actions=list(range(env.n_actions)), trace_decay=0.9)
    for episode in range(100):
        '''
        state = env.reset(episode)
        action = q_table.choose_action(str(state))

        while True:
            env.render()
            state_, reward, done = env.step(action)

            action_ = q_table.choose_action(str(state_)) 
            q_table.learn(str(state), action, reward, str(state_), action_)
            state = state_
            action = action_

            if done:
                break
        '''
        state = env.reset(episode)
        while True:
            env.render()

            action = q_table.choose_action(str(state))
            state_, reward, done = env.step(action)
            q_table.learn(str(state), action, reward, str(state_))

            state = state_
            
            if done:
                break

    
    print('game over')
    env.destroy()


def run_DQN(env: Maze):
    dqn  = DeepQNetwork(
        env.n_actions, 
        env.n_features, 
        learning_rate=0.01,
        e_greedy=0.9,
        replace_target_iter=200,
        memory_size=2000)
        
    step = 0
    for episode in range(300):
        state = env.reset(episode)

        while True:
            env.render()

            action = dqn.choose_action(state)
            state_, reward, done = env.step(action)
            dqn.store_transtion(state, action, reward, state_)

            if step > 200 and step % 5 == 0:
                dqn.learn()

            state = state_

            if done:
                break
            
            step += 1
    
    print('game over')
    env.destroy()


if __name__ == '__main__':
    env = Maze()

    env.after(100, run_DQN, env)
    env.mainloop()