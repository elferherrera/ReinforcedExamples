import gym
import time
import torch
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0', is_slippery=False)

number_states = env.observation_space.n
number_actions = env.action_space.n

# Matrix to store optimal values for actios
Q = torch.zeros((number_states, number_actions), dtype=torch.float)
gamma = 1
episodes = 1000
log = False

timesteps = []
rewards = []
for i_episode in range(episodes):
    state = env.reset()

    step = 0
    while True:
        step += 1

        # Random values to force movement
        random_values = Q[state] + torch.rand(1, number_actions) / 1000

        # Selecting the best action from the actual state
        action = random_values.argmax().item()

        # Calculating the new state, reward for the best action
        new_state, reward, done, info = env.step(action)

        # Updating the Q matrix given the new state
        Q[state, action] = reward + gamma * Q[new_state].max()

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
            break

print(Q)

message = "Completed: {:.0f} Percent: {:0.2f}".format(sum(rewards), sum(rewards)/episodes)

f, ax = plt.subplots(2, 1)
ax[0].plot(timesteps)
ax[0].set_title(message)

ax[1].plot(rewards)
ax[1].set_ylim(0, 2)
plt.show()

env.close()