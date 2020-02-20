import gym
import time
import torch
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0', is_slippery=False)

number_states = env.observation_space.n
number_actions = env.action_space.n

# Matrix to store optimal values for actios
Q = torch.zeros((number_states, number_actions), dtype=torch.float)
gamma = 0.9
alpha = 0.9

egreedy = 0.7
egreedy_final = 0.1
egreedy_decay = 0.999

episodes = 1000
log = False

timesteps = []
rewards = []
egreedy_total = []
for i_episode in range(episodes):
    state = env.reset()

    step = 0
    while True:
        step += 1

        if torch.rand(1).item() > egreedy:
            # Random values to force movement
            random_values = Q[state] + torch.rand(1, number_actions) / 1000

            # Selecting the best action from the actual state
            action = random_values.argmax().item()

        else:
            action = env.action_space.sample()
            egreedy = max(egreedy * egreedy_decay, egreedy_final)

        # Calculating the new state, reward for the best action
        new_state, reward, done, info = env.step(action)

        # Updating the Q matrix given the new state
        previus_knowledge = (1-alpha) * Q[state, action] 
        new_knowledge = alpha * (reward + gamma * Q[new_state].max())

        Q[state, action] = previus_knowledge + new_knowledge

        if log:
            # Rendering the environment
            env.render()

            message = ("Action: {}\t New State: " 
                    "{}\t State: {}\t Step: {}\t Episode: {}").format(
                        action, new_state, state, step, i_episode)

            print(message)
            print(Q)
            time.sleep(0.3)

        # The new state is the actual state
        state = new_state

        if done:
            if log:
                print("Episode finished after {} timesteps".format(step))

            timesteps.append(step)
            rewards.append(reward)
            egreedy_total.append(egreedy)

            break

print(Q)

message = "Completed: {:.0f} Percent: {:0.2f}".format(sum(rewards), sum(rewards)/episodes)

f, ax = plt.subplots(3, 1)
ax[0].plot(timesteps)
ax[0].set_title(message)

ax[1].plot(rewards)
ax[1].set_ylim(0, 2)

ax[2].plot(egreedy_total)
ax[2].set_ylim(0, 1)
plt.show()

env.close()