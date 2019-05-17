import gym

import rs_optimize
import numpy as np
np.set_printoptions(suppress=True)

env = gym.make('BipedalWalker-v2')
# env.seed(0)
# np.random.seed(0)

obs_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
#action_space = env.action_space.n
obs = env.reset()
class Model:
    def __init__(self):
        self.weights = np.zeros(shape=(obs_space,action_space))
    def action(self,obs):
        return np.matmul(obs,self.weights)


model = Model()

def soft_max(X):
    return np.exp(X )/ np.sum(np.exp(X))

def pick_action(obs, weights):

    model.weights = np.array(weights).reshape(obs_space,action_space)
    a = model.action(obs.reshape(1,-1))
    #a = soft_max(a[0])
    #a = np.argmax(a)
    return a[0]


def run_environment(env, weights, steps=1000, render=True, average=1):
    best_reward = []
    for i in range(average):
        step = 0
        reward_sum = 0
        obs = env.reset()
        # action_space = env.action_space.n

        while step < steps:
            if render:
                env.render()
            action = pick_action(obs, weights)

            obs, reward, done, _ = env.step(action)

            reward_sum += reward
            step += 1
            if done:

                break
        best_reward.append(reward_sum)
    assert not np.isnan(np.average(best_reward)), "return is NaN"
    return np.average(best_reward)

steps = 500

max_attempts = 1e3

def f(weights):

    reward_sum = -1*run_environment(env, weights, steps, render=False)
    #assert not np.isnan(reward_sum), "return is NaN"
    return reward_sum

import test_functions
issue = test_functions.Issue(f,obs_space*action_space)

def run(worker):
    # np.random.seed()
    print('Start Worker:', worker)

    best_weights = model.weights
    best_score = 0
    episode = 0
    reward_sum = 0

    while episode < max_attempts and best_score < steps:
        # print(worker,end="") if episode % 500 != 0 else print()
        #         if episode % 100 == 0:
        #             run_environment(env,best_weights,steps,render=True)

        reward_sum = run_environment(env, steps, render=False)

        if reward_sum > best_score:
            best_score = reward_sum
            best_weights = model.weights
            print(f'\nWorker {worker} reporting new score', best_score, episode)


        else:
            model.weights = best_weights


        episode += 1
    print(f'End Worker {worker} after {episode} with {best_score}')
    return best_weights

#results = run(0)

rs_optimize.optimize(issue,0.1,100000,5)