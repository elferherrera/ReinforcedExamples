import gym
import time
import matplotlib.pyplot as plt

#env = gym.make('CartPole-v0')
env = gym.make('FrozenLake-v0', is_slippery=False)

episodes = 100
timesteps = []
for i_episode in range(episodes):
    observation = env.reset()

    for t in range(100):
        env.render()

        action = env.action_space.sample()
        new_observation, reward, done, info = env.step(action)

        print("Action: {}\t Old State: {}\t New state: {}\t Info: {}".format(action, observation, new_observation, info))

        observation = new_observation

        time.sleep(1)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            timesteps.append(t)
            break

plt.plot(timesteps)
plt.show()

env.close()