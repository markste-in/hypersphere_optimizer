import gym
import ray
import rs_optimize
import numpy as np
import test_functions
import multiprocessing

cpu_count = multiprocessing.cpu_count()

env = gym.make('BipedalWalker-v2')
# env.seed(0)
# np.random.seed(0)

obs_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
# action_space = env.action_space.n
obs = env.reset()


class Model:
    def __init__(self):
        self.weights = np.zeros(shape=(obs_space, action_space))

    def action(self, obs):
        return np.matmul(obs, self.weights)


model = Model()


def soft_max(X):
    return np.exp(X) / np.sum(np.exp(X))


def pick_action(obs, weights):
    model.weights = np.array(weights).reshape(obs_space, action_space)
    a = model.action(obs.reshape(1, -1))
    # a = soft_max(a[0])
    # a = np.argmax(a)
    return a[0]


def run_environment(env, weights, steps=500, render=False, average=1):
    best_reward = [single_env_run(env, weights, steps=steps, render=render) for _ in range(average)]

    return np.average(best_reward)


def single_env_run(env, weights, steps=500, render=False):
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
    return reward_sum


steps = 500

max_attempts = 1e3


@ray.remote
def f(weights):
    reward_sum = -1 * run_environment(env, weights, steps, render=False, average=3)

    return reward_sum


issue = test_functions.Issue(f, obs_space * action_space)

print('starting rays with', cpu_count, 'CPUs')
ray.init(num_cpus=cpu_count, local_mode=False, include_webui=True)

file = "bipedalwalker_weights.p"

bw,_ = rs_optimize.optimize(issue, local_stepSize=0.1, max_episodes=10000, N=2, filename=file)
run_environment(env, bw, steps=steps, render=True, average=1)
