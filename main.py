import random
import numpy as np
import pickle
import os
import time

LEARNING_RATE = 0.2
DISCOUNT_RATE = 0.9

class Board:
    def __init__(self):
        self.state = np.zeros(9, dtype=np.int8)
        self.turn = 0

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
        self.turn = random.randint(1,2)
        return self.state

    def random_action(self):
        return np.random.choice(np.where(self.state == 0)[0])

    def step(self, action):
        self.state[action] = self.turn
        victory = self.terminal()
        return self.state, self.reward(victory), victory != 0, victory # state, reward, done , victory

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


class Player:
    def __init__(self):
        self.q_table = {}

    def q_action(self, state_hash, state):
        valid_indeces = np.where(state == 0)[0]
        if state_hash in self.q_table:
            ms = np.max(self.q_table[state_hash][valid_indeces])
        else:
            self.q_table[state_hash] = np.zeros(9) 
            return np.random.choice(valid_indeces)
        return np.random.choice(np.where(self.q_table[state_hash] == ms)[0][np.isin(np.where(self.q_table[state_hash] == ms)[0], valid_indeces)])


    def update_q_table(self, state_hash, action, reward, new_state_hash):
        posterior = LEARNING_RATE * reward if new_state_hash not in self.q_table else LEARNING_RATE * (reward + DISCOUNT_RATE * np.max(self.q_table[new_state_hash].values()))
        if state_hash in self.q_table:
            self.q_table[state_hash][action] = self.q_table[state_hash][action] * (1 - LEARNING_RATE) + posterior
        else:
            self.q_table[state_hash] = np.zeros(9)
            self.q_table[state_hash][action] = posterior
        return self.q_table[state_hash][action]

    def save_policy(self, player):
        with open('policy_' + player, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load_policy(self, player):
        with open('policy_' + player, 'rb') as f:
            self.q_table = pickle.load(f)
        



def train(num_episodes):
    env = Board()
    exploration_rate = 0.7
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 1/num_episodes
    j1 = Player()
    j2 = Player()
    print('Training...')
    for episode in range(num_episodes):
        print('Episode: ', episode, end='\r')
        done = False
        state = env.reset()
        state_hash = str(state)
        states_list = []
        while not done:
            if env.turn == 1:
                player = j1
            else:
                player = j2

            # choose action
            if random.uniform(0,1) > exploration_rate:
                action = player.q_action(state_hash, state)
            else:
                action = env.random_action()
            
            # take action
            new_state, _, done, victory = env.step(action)
            states_list.append((state_hash, action))
            new_state_hash = str(new_state)
            state_hash = new_state_hash
            state = new_state

            # update q tables
            if done == True:
                turn = env.turn - 1
                rewards = [0,0]
                if victory != -1:
                    rewards[turn] = 1
                    rewards[abs(1-turn)] = -1 
                else:
                    rewards = [0,0]
                for i, (state_hash, action) in enumerate(reversed(states_list)):
                    t = abs(turn-i) % 2
                    v = player.update_q_table(state_hash, action, rewards[t], new_state_hash)
                    rewards[t] = v
                    player = j1 if player == j2 else j2
                break
            env.turn = 3 - env.turn
            
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode) #     print("Episode: ", len(rewards_all_episodes), " Reward: ", sum(rewards_all_episodes[int((len(rewards_all_episodes))-num_episodes/1000):int(len(rewards_all_episodes)-1)])/(num_episodes/1000), " Exploration rate: ", exploration_rate, end='\r')
    j1.save_policy('j1')
    j2.save_policy('j2')
    time.sleep(1)
    print()
    print('Training finished')

def play(games):
    env = Board()
    for _ in range(games):
        done = False
        state = env.reset()
        state_hash = str(state)
        j1 = Player()
        j1.load_policy('j1')

        while not done:
            if env.turn == 1:
                valid_indeces = np.where(state == 0)[0]
                ms = np.max(j1.q_table[state_hash][valid_indeces])
                action = np.random.choice(np.where(j1.q_table[state_hash] == ms)[0][np.isin(np.where(j1.q_table[state_hash] == ms)[0], valid_indeces)])
                new_state, _, done, victory = env.step(action)
                new_state_hash = str(new_state)
                state_hash = new_state_hash
                state = new_state
                env.turn = 3 - env.turn
            
            elif env.turn == 2:
                os.system('clear||cls')
                print(env)
                valid_indeces = np.where(state == 0)[0]
                action = 10
                while (action-1) not in valid_indeces:
                    action = int(input('Enter a number (1-9): '))
                
                state, _, done, victory = env.step(action-1)
                state_hash = str(state)
                env.turn = 3 - env.turn
                os.system('clear||cls')
                print(env)
                time.sleep(1.2)
                
        
        print(env)
        if victory == 1:
            print('\n\nCOMPUTER WINS\n')
        elif victory == 2:
            print('\n\nYOU WIN\n')
        else:
            print('\n\nDRAW\n')
        input('Press enter to continue')
                
if __name__ == '__main__':
    q_table = train(50000)
    games = 5
    play(games) 