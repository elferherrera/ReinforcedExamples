import math
import gym
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda") if use_cuda else torch.device("cpu")

env = gym.make("CartPole-v0")

random_seed = 0
env.seed(random_seed)
torch.manual_seed(random_seed)

# Matrix to store optimal values for actios
gamma = 0.99
alpha = 1

egreedy = 1
egreedy_final = 0.02
egreedy_decay = 500

def calculate_epsilon(steps):
    epsilon = (egreedy_final + (egreedy - egreedy_final) * 
                math.exp(-1. * steps / egreedy_decay))

    return epsilon

log = False

###########################
# Network params
learning_rate = 0.01
episodes = 500

inputs = env.observation_space.shape[0]
output = env.action_space.n
hidden_layer = 64

class Q_matrix(nn.Module):
    def __init__(self):
        super(Q_matrix, self).__init__()
        self.linear1 = nn.Linear(inputs, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, output)


    def forward(self, state):
        output = torch.relu(self.linear1(state))
        output = self.linear2(output)

        return output


class Q_agent(object):
    def __init__(self):
        self.q_nn = Q_matrix()
        # Loss function
        self.loss_func = nn.MSELoss()
        # Function optimizer
        self.optimizer = optim.Adam(params=self.q_nn.parameters(), lr=learning_rate)


    def select_action(self, state, epsilon):

        if torch.rand(1).item() > epsilon:
            with torch.no_grad():
                state = torch.Tensor(state).to(device)
                actions_nn = self.q_nn(state)

                action = actions_nn.argmax().item()

        else:
            action = env.action_space.sample()

        return action


    def optimize(self, state, action, new_state, reward, done):
        state = torch.Tensor(state).to(device)
        new_state = torch.Tensor(new_state).to(device)
        reward = torch.Tensor([reward]).to(device)

        if done:
            target_value = reward

        else:
            new_state_values = self.q_nn(new_state).detach()
            max_new_state_values = new_state_values.max()
            target_value = reward + gamma * max_new_state_values

        predicted_value = self.q_nn(state)[action].unsqueeze(0)

        loss = self.loss_func(predicted_value, target_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return None
###########################
 
q_agent = Q_agent()

timesteps = []
rewards = []
total_steps = 0
for i_episode in range(episodes):
    state = env.reset()

    step = 0
    while True:
        step += 1
        total_steps += 1

        epsilon = calculate_epsilon(total_steps)

        action = q_agent.select_action(state, epsilon)

        # Calculating the new state, reward for the best action
        new_state, reward, done, info = env.step(action)

        # Optimize q_nn
        q_agent.optimize(state, action, new_state, reward, done)


        if log:
            # Rendering the environment
            env.render()

            message = ("Action: {}\t New State: " 
                    "{}\t Reward: {}\t Step: {}\t Episode: {}").format(
                        action, new_state, reward, step, i_episode)

            print(message)
            time.sleep(0.3)

        # The new state is the actual state
        state = new_state

        if done:
            if log:
                print("Episode finished after {} timesteps".format(step))

            timesteps.append(step)
            break

message = "Total steps in episodes: {:.0f}".format(sum(timesteps))

plt.plot(timesteps)
plt.title(message)
plt.show()

env.close()