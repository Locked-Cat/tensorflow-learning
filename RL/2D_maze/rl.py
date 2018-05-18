#!/usr/bin/env python3
import numpy as np
import pandas as pd
import tensorflow as tf


class RL(object):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions)

    def check_state_exists(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )

    def choose_action(self, state):
        self.check_state_exists(state)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.ix[state, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            state_action = state_action.astype('float32')
            action = state_action.idxmax()
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass
  

class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, state, action, reward, state_):
        self.check_state_exists(state_)
        q_predict = self.q_table.ix[state, action]
        if state_ != 'terminal':
            q_target = reward + self.gamma * self.q_table.ix[state_, :].max()
        else:
            q_target = reward

        self.q_table.ix[state, action] += self.learning_rate * (q_target - q_predict)


class SarsaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, state, action, reward, state_, action_):
        self.check_state_exists(state_)
        q_predict = self.q_table.ix[state, action]
        if state_ != 'terminal':
            q_target = reward + self.gamma * self.q_table.ix[state_, action_]
        else:
            q_target = reward
        
        self.q_table.ix[state, action] += self.learning_rate * (q_target - q_predict)


class SarsaLambdaTable(RL):
    def __init__(self, actions, trace_decay, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exists(self, state):
        if state not in self.q_table.index:
            to_be_append = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=state
            )

            self.q_table = self.q_table.append(to_be_append)
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, state, action, reward, state_, action_):
        self.check_state_exists(state_)
        q_predict = self.q_table.ix[state, action]
        if state_ != 'terminal':
            q_target = reward + self.gamma * self.q_table.ix[state_, action_]
        else:
            q_target = reward
        
        error = q_target - q_predict

        self.eligibility_trace.ix[state, :] *= 0
        self.eligibility_trace.ix[state, action] = 1

        self.q_table += self.learning_rate * error * self.eligibility_trace
        self.eligibility_trace *= self.gamma * self.lambda_


class DeepQNetwork:
    def __init__(self,
        n_actions,
        n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=300,
        memory_size=500,
        batch_size=32,
        e_greedy_increment=None,
        output_graph=False
        ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.e_greedy_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.e_greedy_increment  = e_greedy_increment
        self.e_greedy = 0 if e_greedy_increment is not None else self.e_greedy_max
        self.learning_step = 0

        self.memory = pd.DataFrame(np.zeros((self.memory_size, n_features * 2 + 2)))
        self._build_net()
        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter('logs/', graph=self.sess.graph)
        
        self.sess.run(tf.global_variables_initializer())
        self.cost_history = []


    def _build_net(self):
        # eval_net
        self.state = tf.placeholder(tf.float32, shape=[None, self.n_features], name='state')
        self.q_target = tf.placeholder(tf.float32, shape=[None, self.n_actions], name='q_target')

        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initialzer, b_initialzer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(mean=0.0, stddev=0.3), tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initialzer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initialzer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.state, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initialzer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initialzer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_sum(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self.trainer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        # target_net
        self.state_ = tf.placeholder(tf.float32, [None, self.n_features], name='state_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initialzer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initialzer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.state_, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initialzer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initialzer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transtion(self, state, action, reward, state_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((state, [action, reward], state_))

        index = self.memory_counter % self.memory_size
        self.memory.iloc[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, state):
        state = state[np.newaxis, :]

        if np.random.uniform() < self.e_greedy:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.state: state})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def _replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

    def learn(self):
        if self.learning_step % self.replace_target_iter == 0:
            self._replace_target_params()
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.state_: batch_memory.iloc[: -self.n_features:],
                self.state: batch_memory.iloc[: :self.n_features]
            }
        )

        q_target  = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[: self.n_features + 1]
        q_target[batch_index, eval_act_index] = reward + self.reward_decay * np.max(q_next, axis=1)

        _, self.cost = self.sess.run(
            [self.trainer, self.loss],
            feed_dict={
                self.state: batch_memory.iloc[:, :self.n_features],
                self.q_target: q_target
                }
            )
        
        self.cost_history.append(self.cost)
        self.e_greedy  = self.e_greedy + self.e_greedy_increment if self.e_greedy < self.e_greedy_max else 0
        self.learning_step += 1       
    
    def plot_cost(self):
        import matplotlib.pyplot as plt 
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.show()
