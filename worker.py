import gym
import ray
import rs_optimize
import numpy as np
import test_functions
import pickle
import multiprocessing

cpu_count = multiprocessing.cpu_count()


env = gym.make('BipedalWalker-v2')
env.seed(0)
np.random.seed(0)

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
@ray.remote
def f(weights):

    reward_sum = -1*run_environment(env, weights, steps, render=False)

    return reward_sum


issue = test_functions.Issue(f,obs_space*action_space)


ray.init(num_cpus=cpu_count, local_mode=False, include_webui=True)

SP = np.random.uniform(-1,1,issue.dim)

while True:
    SP = rs_optimize.optimize(issue,0.1,200,20,SP=SP)
    pickle.dump(SP, open("bipedalwalker_weights.p", "wb"))
    run_environment(env, SP, steps=1000, render=True)
