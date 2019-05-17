import numpy as np
import ray
import HelperFunctions

search_range = [-1,1]

import HelperFunctions


def points_on_sphere(dim=3, N = 10):
    return np.array([point_on_sphere(dim) for i in range(N)])

def point_on_sphere(dim=3):
    dim = int(dim)
    xyz = np.random.normal(0, 1,dim)
    inv_l = np.array(1. / np.sqrt(np.sum(np.square(xyz))))
    return inv_l * xyz

def points_in_cloud(dim = 3,N=10):
    return np.array([np.random.random(int(dim)) for i in range(N)])

@HelperFunctions.t_decorator
def optimize(issue, local_stepSize = 1., max_episodes = 100, N = 5, SP = None):
    f = issue.f
    Dim = issue.dim
    dynamic_stepSize = local_stepSize


    SP = np.random.uniform(search_range[0],search_range[1],Dim) if type(SP) == type(None) else SP
    best_value = ray.get(f.remote(SP))

    #x, y, z = zip(*xyz)
    print('starting here:',SP)
    print('value:',best_value)
    failed_to_improve = 0
    best_point_on_sphere = None

    for i in range(max_episodes):

        hyperSp = points_on_sphere(dim=Dim,N=N)
        dynamic_stepSize = np.random.choice(np.linspace(local_stepSize, 10.*local_stepSize,10))


        hyperSp = np.array(hyperSp)

        candidates = SP + local_stepSize * hyperSp
        dynamic_candidates = SP + dynamic_stepSize*hyperSp

        eval_cand=ray.get([ f.remote(can) for can in candidates])
        dynamic_eval_cand = ray.get([f.remote(dyn) for dyn in dynamic_candidates])

        min_val_local = np.min(eval_cand)
        min_val_global = np.min(dynamic_eval_cand)

        local_global_success = np.argmin([min_val_local,min_val_global])
        delta = best_value - np.min([min_val_local, min_val_global])
        assert not np.isnan(delta), "detla is NaN"
        if (np.min([min_val_local,min_val_global]) <best_value):


            if min_val_local<min_val_global:
                SP = candidates[np.argmin(eval_cand)]
            else:
                SP = dynamic_candidates[np.argmin(dynamic_eval_cand)]
            best_point_on_sphere = hyperSp[np.argmin(eval_cand)]

            if (delta < 1e-3):
                failed_to_improve += 1
                print('.',end='')
            else:
                failed_to_improve = 0

                if local_global_success and (dynamic_stepSize > local_stepSize):
                    local_stepSize = dynamic_stepSize
                # else:
                #     dynamic_stepSize = np.random.randint(0,search_range[1]-search_range[0])/2.

            best_value = np.min([min_val_local,min_val_global])
            print('\n', i, '[', failed_to_improve, ']', 'improved!->', best_value, 'delta:', delta, 'with',
                  'global step' if local_global_success else 'local step', '### SZ:', local_stepSize, dynamic_stepSize, ' ### avg hs:', np.average(hyperSp), 'avg SP:', np.average(np.abs(SP)))

            #print(best_point_on_sphere)


        else:
            print('x',end='')
            failed_to_improve +=1
            if (failed_to_improve > 5 and failed_to_improve % 5 ==0):
                local_stepSize /= 2.
                print('\n', i, '[', failed_to_improve, ']','failed to improve for more than 5 steps -> reducing step size to', local_stepSize)
                print('\n', i, '[', failed_to_improve, ']', '->', best_value, 'delta:', delta, 'with',
                      'global step' if local_global_success else 'local step', 'SZ:', local_stepSize, dynamic_stepSize)

            if (failed_to_improve > 50 and failed_to_improve % 50 == 0):
                local_stepSize = np.random.random()
                print('\n', i, '[', failed_to_improve, ']','failed to improve for more than 50 steps -> resetting step size to', local_stepSize)
                print('\n', i, '[', failed_to_improve, ']', '->', best_value, 'delta:', delta, 'with',
                      'global step' if local_global_success else 'local step', 'SZ:', local_stepSize, dynamic_stepSize)
            # print(i, '[', failed_to_improve, ']', SP, 'not improved!->', best_value, 'delta:', delta, 'with',
            #       'global step' if local_global_success else 'local step', 'SZ:', local_stepSize, dynamic_stepSize)

    return SP


