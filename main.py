import random
import numpy as np


class Board:
    def __init__(self):
        self.state = np.zeros(9, dtype=np.int8)
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
        self.state = np.zeros(9, dtype=np.int8)
        return self.state

    def random_action(self):
        return np.random.choice(np.where(self.state == 0)[0])

    def q_action(self, q_table, state_hash):
        valid_indeces = np.where(state == 0)[0]
        if state_hash in q_table:
            ms = np.max(q_table[state_hash][valid_indeces])
        else:
            q_table[state_hash] = np.zeros(9) 
            return np.random.choice(valid_indeces)
        return np.random.choice(np.where(q_table[state_hash] == ms)[0][np.isin(np.where(q_table[state_hash] == ms)[0], valid_indeces)])
        

    def step(self, action):
        if self.state[action] != 0:
            raise Exception('Invalid action')
        self.state[action] = self.turn
        self.turn = 3 - self.turn
        victory = self.terminal()
        return self.state, self.reward(victory), victory != 0, victory

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
        if self.turn == 2:
            if terminal == 1:
                return 1
            elif terminal == 2:
                return -1
            elif terminal == 0:
                return 0
        else:
            if terminal == 1:
                return -1
            elif terminal == 2:
                return 1
            elif terminal == 0:
                return 0
        return 0


def update_q_table(q_table, state_hash, action, reward, new_state_hash):
    posterior = learning_rate * reward if new_state_hash not in q_table else learning_rate * (reward + discount_rate * np.max(q_table[new_state_hash].values()))
    if state_hash in q_table:
        q_table[state_hash][action] = q_table[state_hash][action] * (1 - learning_rate) + posterior
    else:
        q_table[state_hash] = np.zeros(9)
        q_table[state_hash][action] = posterior
    return q_table[state_hash][action]


q_table = {}


env = Board()

num_episodes = 6000000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.9

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.000001

rewards_all_episodes = []

for episode in range(num_episodes):
    done = False
    state = env.reset()
    
    state_hash = str(state)
    rewards_current_episode = 0
    states_list = []
    for step in range(max_steps_per_episode):
        if random.uniform(0,1) > exploration_rate:
            action = env.q_action(q_table, state_hash)
        else:
            action = env.random_action()
        new_state, reward, done, victory = env.step(action)
        states_list.append((state_hash, action))
        # Guardarse el estado
        # cuando llegas al done backpropagar el error
        new_state_hash = str(new_state)
        # update_q_table(q_table, state_hash, action, reward, new_state_hash)
        state_hash = new_state_hash
        rewards_current_episode += reward
        if done == True:
            turn = 2 - env.turn # 1
            rewards=[0,0]
            if victory != -1:
                rewards[turn]=1 #[0,1]
            else:
                rewards = [0.1,0.1]
            # Feed rewards backwards
            # al que gana le das el reward entero 
            for i, (state_hash, action) in enumerate(reversed(states_list)):
                v=update_q_table(q_table, state_hash, action, rewards[abs(turn-i)%2], new_state_hash)
                rewards[abs(turn-i)%2]=v
            break
                

    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    rewards_all_episodes.append(rewards_current_episode)
    if len(rewards_all_episodes) % (num_episodes/100) == 0:
        print("Episode: ", len(rewards_all_episodes), " Reward: ", sum(rewards_all_episodes[int((len(rewards_all_episodes))-num_episodes/100):int(len(rewards_all_episodes)-1)])/(num_episodes/100), " Exploration rate: ", exploration_rate)

rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), 10)
count = num_episodes/10
print("Average reward per thousand episodes")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/(num_episodes/10))))
    count += num_episodes/10

for i in range(9):
    done = False
    state = env.reset()
    state_hash = state.tobytes()
    rewards_current_episode = 0
    turno = random.randint(0,1)
    while not done:
        if turno == 1:
            valid_indeces = np.where(state == 0)[0]
            # quiero el máximo de los valores de q_table[state_hash] en los índices válidos
            # si hay varios máximos, elijo uno que esté en valid_indeces
            ms = np.max(q_table[state_hash][valid_indeces])
            # the suggestion is wrong
            # action = np.random.choice(np.where(q_table[state_hash] == ms)[0]) because there can be more than one maximum and we want to choose one of the valid ones
            action = np.random.choice(np.where(q_table[state_hash] == ms)[0][np.isin(np.where(q_table[state_hash] == ms)[0], valid_indeces)])
            new_state, reward, done, victory = env.step(action)
            new_state_hash = new_state.tobytes()
            update_q_table(q_table, state_hash, action, reward, new_state_hash)
            state = new_state
            rewards_current_episode += reward
            turno = 1 - turno
        
        elif turno == 0 and not done:
            print(env)
            action = int(input('mete el puto numero:'))
            new_state, reward, done, victory = env.step(action)
            state = new_state
            turno = 1 - turno
        if done == True:
            break
        
    
    print(env)
    input()