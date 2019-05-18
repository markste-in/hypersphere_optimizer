from rs_optimize import *
from test_functions import *
import multiprocessing


cpu_count = multiprocessing.cpu_count()

dim = 2

candidateSize = 10

print('starting rays with', cpu_count, 'CPUs')
ray.init(num_cpus=cpu_count, local_mode=False, include_webui=True)

issue = Issue('rastrigin', dim=dim)

optimize(issue, local_stepSize=.001, max_episodes=5000, N=candidateSize)

