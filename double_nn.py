import math
import gym
import time
import random
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

###########################
# Network params
# Matrix to store optimal values for action
gamma = 1

egreedy = 0.9
egreedy_final = 0.01
egreedy_decay = 500

learning_rate = 0.01
episodes = 500

replay_memory_size = 50000
batch_size = 32
update_frequency = 100

inputs = env.observation_space.shape[0]
output = env.action_space.n
hidden_layer = 64

clip_error = False
log = False

def calculate_epsilon(steps):
    epsilon = (egreedy_final + (egreedy - egreedy_final) * 
                math.exp(-1. * steps / egreedy_decay))

    return epsilon

class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0


    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)

        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))


    def __len__(self):
        return len(self.memory)


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
        self.q_nn = Q_matrix().to(device)
        self.target_nn = Q_matrix().to(device)
        # Loss function
        self.loss_func = nn.MSELoss()
        # Function optimizer
        self.optimizer = optim.Adam(params=self.q_nn.parameters(), lr=learning_rate)

        self.update_target_counter = 0


    def select_action(self, state, epsilon):

        if torch.rand(1).item() > epsilon:
            with torch.no_grad():
                state = torch.Tensor(state).to(device)
                actions_nn = self.q_nn(state)

                action = actions_nn.argmax().item()

        else:
            action = env.action_space.sample()

        return action


    def optimize(self):
        if len(memory) < batch_size:
            return None

        state, action, new_state, reward, done = memory.sample(batch_size)

        state = torch.Tensor(state).to(device)
        action = torch.LongTensor(action).to(device)
        new_state = torch.Tensor(new_state).to(device)
        reward = torch.Tensor(reward).to(device)
        done = torch.Tensor(done).to(device)

        new_state_values = self.target_nn(new_state).detach()
        max_new_state_values = new_state_values.max(dim=1).values
        target_value = reward + (1-done) * gamma * max_new_state_values

        predicted_value = self.q_nn(state).gather(1, action.unsqueeze(1))

        loss = self.loss_func(predicted_value, target_value.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        if clip_error:
            for param in self.q_nn.parameters():
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        if self.update_target_counter % update_frequency == 0:
            self.target_nn.load_state_dict(self.q_nn.state_dict())
        
        self.update_target_counter += 1

###########################
 
q_agent = Q_agent()
memory = ExperienceReplay(replay_memory_size)

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

        # Store in memory
        memory.push(state, action, new_state, reward, done)

        # Optimize q_nn
        q_agent.optimize()


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