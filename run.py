from env_heuristic import Environment
from cacheAlgo import CacheAlgo
from dqn.DQN_brain import DeepQNetwork
from utils.plotting import EpisodeStats
import numpy as np
from utils.config import *
import os.path
import pickle
import argparse
from utils.GenRequest import *


def load_file(path, name):
    with open(os.path.join(path, name), 'rb') as f:
        file = pickle.load(f)
        print("success to load the file: %s" % path+name)
    return file

def save_file(path, name, obj):
    with open(os.path.join(path, name), 'wb') as f:
        pickle.dump(obj, f)
    print("Success to generate and save the file: %s" % path+name)

def run(integrated_file):
    # data_list = integrated_file['data_list']
    # requests = integrated_file['requests']
    # zipf = integrated_file['zipf']
    env = integrated_file['environment']
    env.cache_size = cache_size
    print(env.cache_size)

    algo1 = CacheAlgo("proposed", env)
    algo2 = CacheAlgo("optimal", env)
    algo3 = CacheAlgo("random", env)
    algo4 = CacheAlgo("MPC", env)
    env.add_algo(algo1)
    env.add_algo(algo2)
    env.add_algo(algo3)
    env.add_algo(algo4)

    hit_result = [0 for _ in range(len(env.algo_lst))]
    delay_result = [0 for _ in range(len(env.algo_lst))]
    refresh_result = [0 for _ in range(len(env.algo_lst))]

    t = 0

    while t <= end_time:
        env.load_curr_request(t)

        if (t != 0) and (t % update_period == 0):
            print(f'Update cache at time {t}')
            env.update_state()
            env.proactive_caching()
            env.clear()

        hit_lst, delay_lst = env.request()
        for i in range(len(hit_lst)):
            hit_result[i] += hit_lst[i]
            delay_result[i] += delay_lst[i]

        # print(f'---------------time: {t}---------------\nhit result: {hit_result}\ndelay result: {delay_result}\n')
        t += 1

        algo_idx = -1
        for algo in env.algo_lst:
            algo_idx += 1
            # print(algo.algo_name)
            algo.update_fresh()
            algo.update_cache(alpha)
            refresh_result[algo_idx] = algo.activation_cnt

    result = {f'total_request: {env.total_requests}, hit_count: {hit_result}, hit_ratio: {np.array(hit_result) / env.total_requests}, '
              f'total_delay: {delay_result}, refresh_count: {refresh_result}'}
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', action='store_true',
                       help='algorithm name.')
    args = vars(parser.parse_args())

    if os.path.exists(file_path + file_name):
        integrated_file = load_file(file_path, file_name)
        # data_list = file['data_list']
        # requests = file['requests']
        # env = file['environment']
        # zipf = file['zipf']
    else:
        data_list = make_data(num_server, num_data, data_size, life_time)
        zipf = Zipf()
        zipf.set_env(zipf_param, num_data)
        requests = make_request_events(num_server, arrival_rate, end_time, zipf, data_list)
        env = Environment(requests, end_time, update_period, num_server, cache_size, arrival_rate, data_list)
        integrated_file = {'data_list': data_list, 'requests': requests, 'zipf': zipf, 'environment': env}

        save_file(file_path, file_name, integrated_file)

    run(integrated_file)










