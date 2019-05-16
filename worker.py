import gym
import multiprocessing
import tensorflow as tf

import numpy as np
np.set_printoptions(suppress=True)

env = gym.make('BipedalWalker-v2')
env.seed(0)
np.random.seed(0)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(4,input_shape=(24,),activation='tanh'))
model.build()

obs = env.reset()
model.predict(obs.reshape(1,-1))

def pick_action(obs):
    a = model.predict(obs.reshape(1,-1))
    return a[0]

def generate_mutation(weights):
    gamma = np.random.choice((np.logspace(0,1,num=200)-0.999)**2)
    for i, w in enumerate(weights[0]):
        if np.random.randint(0,2): # full mutation
            weights[0][i] = gamma * np.random.random(w.shape)*2-1
        else: #spot mutation
            spot = np.random.choice(weights[0][i].size) # spot that gets mutated
            rnd = gamma * np.random.rand()*2-1 #mutation value
            rvl = weights[0][i].ravel() #mutate
            rvl[spot] = rnd #mutate
            weights[0][i]= rvl.reshape(weights[0][i].shape) #write mutation
    return weights

def run_environment(env, weights, steps=1000, render=False, average=1):
    best_reward = []
    for i in range(average):
        step = 0
        reward_sum = 0
        obs = env.reset()
        # action_space = env.action_space.n

        while step < steps:
            if render: env.render()
            action = pick_action(obs)

            obs, reward, done, _ = env.step(action)

            reward_sum += reward
            step += 1
            if done:
                best_reward.append(reward_sum)
                break

    return np.average(best_reward)

steps = 500
obs_space = env.observation_space.shape[0]
max_attempts = 1e9

def run(worker):
    # np.random.seed()
    print('Start Worker:', worker)

    weights = [layer.get_weights() for layer in model.layers]
    best_weights = weights
    best_score = -2000
    episode = 0
    reward_sum = 0

    while episode < max_attempts and best_score < steps:
        # print(worker,end="") if episode % 500 != 0 else print()
        #         if episode % 100 == 0:
        #             run_environment(env,best_weights,steps,render=True)

        reward_sum = run_environment(env, weights, steps, render=True)

        if reward_sum > best_score:
            best_score = reward_sum
            best_weights = weights
            print(f'\nWorker {worker} reporting new score', best_score, episode)

        else:
            weights = best_weights

        weights = generate_mutation(weights)
        # print( weights)
        for i, layer in enumerate(model.layers): model.set_weights(weights[i])

        episode += 1
    print(f'End Worker {worker} after {episode} with {best_score}')
    return best_weights

results = run(0)