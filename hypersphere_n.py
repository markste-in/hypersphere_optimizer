from rs_optimize import *
from test_functions import *
import multiprocessing


cpu_count = multiprocessing.cpu_count()

dim = 10

candidateSize = 10

print('starting rays with', cpu_count, 'CPUs')
ray.init(num_cpus=cpu_count, local_mode=False, include_webui=True)

issue = Issue('sphere', dim=dim)

optimize(issue, local_stepSize=.1, max_episodes=500, N=candidateSize)

