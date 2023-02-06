import random
import numpy as np


class Board:
    def __init__(self):
        self.state = np.zeros(9)
        self.turn = 1

    def __str__(self):
        XO = {0: '   ', 1: ' X ', 2: ' O '}
        str = '╔' + ('═'*3 + '╦')*2 + '═'*3 + '╗'
        for idx, elem in enumerate(self.state):
            if idx % 3 == 0:
                str += '\n║'
            str += XO[int(elem)] + '║'
            if idx % 3 == 2 and idx != 8:
                str += '\n╠' + ('═'*3 + '╬')*2 + '═'*3 + '╣'
        str += '\n╚' + ('═'*3 + '╩')*2 + '═'*3 + '╝'
        return str

    def reset(self):
        self.state = np.zeros(9)
        return self.state

    def random_action(self):
        return np.random.choice(np.where(self.state == 0)[0])

    def step(self, action):
        if self.state[action] != 0:
            raise Exception('Invalid action')
        self.state[action] = self.turn
        self.turn = 3 - self.turn
        victory = self.terminal()
        return self.state, self.reward(victory), victory != 0

    def terminal(self):
        # 0: not terminal
        # 1: player 1 wins
        # 2: player 2 wins 
        # -1: draw
        for i in range(3):
            if self.state[i*3] == self.state[i*3+1] == self.state[i*3+2] != 0:
                return 2 if self.state[i*3] == 2 else 1
            if self.state[i] == self.state[i+3] == self.state[i+6] != 0:
                return 2 if self.state[i] == 2 else 1
        if self.state[0] == self.state[4] == self.state[8] != 0:
            return 2 if self.state[0] == 2 else 1
        if self.state[2] == self.state[4] == self.state[6] != 0:
            return 2 if self.state[2] == 2 else 1
        if 0 not in self.state:
            return -1
        return 0

    def reward(self, terminal):
        if terminal == 1:
            return 1
        elif terminal == 2:
            return -1
        else:
            return 0



def update_q_table(q_table, state_bytes, action, reward, new_state_bytes):
    posterior = learning_rate * reward if new_state_bytes not in q_table else learning_rate * (reward + discount_rate * np.max(q_table[new_state_bytes].values()))
    if state_bytes in q_table:
        q_table[state_bytes][action] = q_table[state_bytes][action] * (1 - learning_rate) + posterior
    else:
        q_table[state_bytes] = np.zeros(9)
        q_table[state_bytes][action] = posterior


q_table = {}


env = Board()

num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []

for episode in range(num_episodes):
    done = False
    state = env.reset()
    state_bytes = state.tobytes()
    rewards_current_episode = 0
    for step in range(max_steps_per_episode):
        if random.uniform(0,1) > exploration_rate:
            ms = np.max(q_table[state_bytes][np.where(state == 0)])
            if isinstance(ms, np.ndarray):
                for i in ms:
                    a = np.where(q_table[state_bytes] == i)[0][0]
                    if state[a] == 0:
                        action = a
                        break
            else:
                action = np.where(q_table[state_bytes] == ms)[0][0]
        else:
            action = env.random_action()
        new_state, reward, done = env.step(action)
        new_state_bytes = new_state.tobytes()
        update_q_table(q_table, state_bytes, action, reward, new_state_bytes)
        state = new_state
        rewards_current_episode += reward
        if done == True:
            break
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    rewards_all_episodes.append(rewards_current_episode)

rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/1000)
count = 1000
print("Average reward per thousand episodes")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

for i in range(9):
    done = False
    state = env.reset()
    state_bytes = state.tobytes()
    rewards_current_episode = 0
    for step in range(max_steps_per_episode):
        ms = np.max(q_table[state_bytes][np.where(state == 0)])
        if isinstance(ms, np.ndarray):
            for i in ms:
                a = np.where(q_table[state_bytes] == i)[0][0]
                if state[a] == 0:
                    action = a
                    break
        else:
            action = np.where(q_table[state_bytes] == ms)[0][0]
        new_state, reward, done = env.step(action)
        new_state_bytes = new_state.tobytes()
        update_q_table(q_table, state_bytes, action, reward, new_state_bytes)
        state = new_state
        print(env)
        input()
        rewards_current_episode += reward
        if done == True:
            break
    
    
    
    input()