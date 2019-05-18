import numpy as np
import ray

search_range = [-1,1]

import HelperFunctions


def points_on_sphere(dim=3, N = 10):
    return np.array(ray.get([point_on_sphere.remote(dim) for i in range(N)]))

@ray.remote
def point_on_sphere(dim=3):
    dim = int(dim)
    xyz = np.random.normal(0, 1,dim)
    inv_l = np.array(1. / np.sqrt(np.sum(np.square(xyz))))
    return inv_l * xyz

def points_in_cloud(dim = 3,N=10):
    return np.array([np.random.random(int(dim)) for i in range(N)])

@HelperFunctions.t_decorator
def optimize(issue, local_stepSize = 1., max_episodes = 100, N = 5, starting_weights = None):
    f = issue.f
    Dim = issue.dim

    SP = np.random.uniform(search_range[0],search_range[1],Dim) if type(starting_weights) == type(None) else starting_weights

    best_value = ray.get(f.remote(SP))

    #x, y, z = zip(*xyz)
    print('starting here:',SP[:5])
    print('value:',best_value)
    failed_to_improve = 0


    for i in range(max_episodes):

        hyperSp = points_on_sphere(dim=Dim,N=N)
        global_stepSize = np.random.choice(np.linspace(local_stepSize, 10.*local_stepSize,10))


        hyperSp = np.array(hyperSp)

        local_candidates = np.clip(SP + local_stepSize * hyperSp,search_range[0],search_range[1])
        global_candidates = np.clip(SP + global_stepSize*hyperSp,search_range[0],search_range[1])


        local_eval_cand=ray.get([ f.remote(can) for can in local_candidates])
        global_eval_cand = ray.get([f.remote(dyn) for dyn in global_candidates])

        min_val_local = np.min(local_eval_cand)
        min_val_global = np.min(global_eval_cand)

        local_global_success = np.argmin([min_val_local,min_val_global])
        delta = best_value - np.min([min_val_local, min_val_global])
        assert not np.isnan(delta), "detla is NaN"
        if (np.min([min_val_local,min_val_global]) <best_value):


            if min_val_local<min_val_global:
                SP = local_candidates[np.argmin(local_eval_cand)]
            else:
                SP = global_candidates[np.argmin(global_eval_cand)]

            if (delta < 1e-3):
                failed_to_improve += 1
                print('.',end='')
            else:
                failed_to_improve = 0

                #if local_global_success and (global_stepSize > local_stepSize):
                local_stepSize = global_stepSize


            best_value = np.min([min_val_local,min_val_global])
            print('\n', i, '[', failed_to_improve, ']', 'improved!->', best_value, 'delta:', delta, 'with',
                  'global step' if local_global_success else 'local step', '### SZ:', local_stepSize, global_stepSize, ' ### avg hs:', np.average(hyperSp), 'avg SP:', np.average(np.abs(SP)))



        else:
            print('x',end='')
            failed_to_improve +=1
            if (failed_to_improve > 5 and failed_to_improve % 5 ==0):
                local_stepSize /= 2.
                print('\n', i, '[', failed_to_improve, ']','failed to improve for more than 5 steps -> reducing step size to', local_stepSize)
                print('\n', i, '[', failed_to_improve, ']', '->', best_value, 'delta:', delta, 'with',
                      'global step' if local_global_success else 'local step', 'SZ:', local_stepSize, global_stepSize)

            if (failed_to_improve > 50 and failed_to_improve % 50 == 0):
                local_stepSize = np.random.random()
                print('\n', i, '[', failed_to_improve, ']','failed to improve for more than 50 steps -> resetting step size to', local_stepSize)
                print('\n', i, '[', failed_to_improve, ']', '->', best_value, 'delta:', delta, 'with',
                      'global step' if local_global_success else 'local step', 'SZ:', local_stepSize, global_stepSize)

    return SP


