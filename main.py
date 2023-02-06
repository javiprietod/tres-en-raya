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
        return self.state, self.reward(), self.victory()
    
    def victory(self):
        for i in range(3):
            if self.state[i*3] == self.state[i*3+1] == self.state[i*3+2] != 0:
                return True
            if self.state[i] == self.state[i+3] == self.state[i+6] != 0:
                return True
        if self.state[0] == self.state[4] == self.state[8] != 0:
            return True
        if self.state[2] == self.state[4] == self.state[6] != 0:
            return True
        return False




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
        exploration_rate_threshold = random.uniform(0,1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state_bytes][action])
        else:
            action = env.random_action()
        new_state, reward, done = env.step(action)
        new_state_bytes = new_state.tobytes()
        if state_bytes in q_table:
            if action in q_table[state_bytes]:
                q_table[state_bytes][action] = q_table[state_bytes][action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state_bytes].values()))
            else:
                q_table[state_bytes][action] = learning_rate * (reward + discount_rate * np.max(q_table[new_state_bytes].values()))
        else:
            q_table[state_bytes] = {}
            q_table[state_bytes][action] = learning_rate * (reward + discount_rate * np.max(q_table[new_state_bytes].values()))
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
    
