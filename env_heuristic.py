import numpy as np
from elements import Server
from utils.config import *
import networkx as nx
from itertools import combinations
import random
from elements import Data
import math
import itertools
import statistics
np.seterr(divide='ignore', invalid='ignore')



# Controller
class Environment:
    def __init__(self, requests, end_T, update_period, num_svr, cache_size, arrival_rate, data_list):
        #data
        self.data_lst = data_list
        self.num_data = len(data_list)
        self.d_size_map = np.zeros(self.num_data, dtype=np.int_)
        self.data_lifetime = np.full(self.num_data, -1, dtype=np.int_)

        self.algo_lst = list()

        self.num_svr = num_svr
        self.cache_size = cache_size
        self.arrival_rate = arrival_rate

        self.end_T = end_T
        self.curr_t = 0
        self.update_period = update_period
        self.requests = requests

        self.svr_lst = list()
        self.lambda_i = np.zeros(self.num_data, dtype=np.float_)

        self.curr_req = list()
        self.pre_req = 0

        self.freshness = np.full((self.num_svr, self.num_data), -1, dtype=np.int_)
        self.cache_mat = np.zeros((self.num_svr, self.num_data), dtype=np.bool_)
        self.lambda_ = np.zeros((self.num_svr, self.num_data), dtype=np.float_)
        # self.wait_queue = np.zeros((self.num_svr, self.num_data), dtype=np.float_)
        self.total_requests = 0

        self.graph = None
        self.rtt_map = np.zeros((self.num_svr, self.num_svr), dtype=np.float_)

        self.create_env()

    def create_env(self):
        print("create env...")
        self.svr_lst = [Server(i, self.cache_size, self.num_data, self) for i in range(self.num_svr)]

        for data in self.data_lst:
            self.d_size_map[data.id] = data.size
            self.data_lifetime[data.id] = data.lifetime

        g = nx.Graph()
        for s in self.svr_lst:
            g.add_node(s.id)
            # self.usable_storage[i] = i.get_usable_storage()

        for u, v in combinations(g, 2):
            g.add_edge(u, v, rtt=random.uniform(0,1)*0.001+0.001)
        self.graph = g

        self.rtt_mapping()
        # for g in self.graph:
        #     nx.draw(g)
        # plt.show()

    def rtt_mapping(self):
        for i in range(self.num_svr):
            for j in range(i+1, self.num_svr):
                length = nx.shortest_path_length(self.graph, i, j, weight='rtt')
                self.rtt_map[i, j] = length
                self.rtt_map[j, i] = length


    def add_algo(self, algo):
        if type(algo).__name__ == 'CacheAlgo':
            self.algo_lst.append(algo)
            print('Success to add algo')
        else:
            print('wrong algo class')

    def load_curr_request(self, t):   # set requests of current time t
        self.curr_req.clear()
        self.curr_t = t
        for r in self.requests:
            if r[0] == t:
                self.curr_req.append(r)
        self.pre_req += len(self.curr_req)
        # print(f'time {t}, request length {len(self.curr_req)}')

    def request(self):
        hit_lst = [0 for _ in range(len(self.algo_lst))]
        delay_lst = [0 for _ in range(len(self.algo_lst))]

        while self.curr_req:
            req = self.curr_req.pop(0)  #(t,svr, data_obj)
            # print("data", req[2].id)
            self.svr_lst[req[1]].request(req[2])
            self.total_requests += 1
            for idx, algo in enumerate(self.algo_lst):
                hit, delay = algo.request(req)
                hit_lst[idx] += hit
                delay_lst[idx] += delay
        return hit_lst, delay_lst

    def clear(self):
        self.pre_req = 0
        for s in self.svr_lst:
            s.clear()

    def proactive_caching(self):
        for algo in self.algo_lst:
            algo.clear()
            algo.proactive_caching()

    # def calc_popularity(self):
    #     popularity = np.zeros(self.num_data, dtype=np.float_)
    #
    #     for s in self.svr_lst:
    #         popularity = np.add(popularity, s.requests_for_data)
    #     popularity = popularity / len(self.curr_req)
    #
    #     return popularity


    def update_state(self):
        l = np.zeros(self.num_data, dtype=np.float_)
        for s in range(self.num_svr):
            lambda_mi = self.svr_lst[s].get_lambda(self.pre_req)
            self.lambda_[s] = lambda_mi
            self.lambda_i = np.add(l, lambda_mi)


    # def calc_trans_time(self):
    #     avg_trans = np.zeros(self.num_svr, dtype=np.float_)
    #     for m in range(self.num_svr):
    #         d_m = 0
    #         d_nm = 0
    #         d_device = 0
    #         for i in range(self.num_data):
    #             d_size = self.get_size(i)
    #             if self.cache_mat[m][i] is True and self.freshness[m][i] > 0:    #data i is cached in source server m
    #                 d_m = d_size/env_params['backhaul']
    #             elif list(np.nonzero(self.cache_mat[:, i])[0]): #data i is cached in neighbor server n
    #                 neighbor_svr = list(np.nonzero(self.cache_mat[:, i])[0])
    #                 if self.freshness[m][neighbor_svr[0]] > 0:
    #                     rtt = self.rtt_map[m][neighbor_svr[0]]
    #                     d_nm = d_size * rtt
    #             else:
    #                 R_m = env_params['bandwidth']*math.log2(1+((env_params['power']*env_params['channel gain']) / (env_params['noise power']*env_params['bandwidth'])))
    #                 d_device = d_size/R_m
    #
    #             avg_trans[m] += self.lambda_[m][i] * (d_m + d_nm + d_device)
    #         # print(self.lambda_[m], avg_trans[m])
    #     # print("Calculate transmission time", avg_trans)
    #     return avg_trans


    # def calc_wait_time(self):
    #     avg_wait = np.zeros(self.num_svr, dtype=np.float_)
    #     for m in range(self.num_svr):
    #         e1 = 0
    #         e2 = 0
    #         for i in range(self.num_data):
    #             service_t = self.get_size(i) / env_params['backhaul']
    #             e2 += pow(service_t, 2) * self.lambda_[m][i]
    #             e1 += service_t * self.lambda_[m][i]
    #         avg_wait[m] = e2 / (2 * (1 - e1))
    #     # print("Calculate waiting time", avg_wait)
    #     return avg_wait
    #
    # def calc_total_time(self, t, w):
    #     tmp = np.add(t, w)
    #     return np.sum(tmp)

    def get_size(self, data):
        if type(data) == Data:
            data = data.id
        elif type(data) == int:
            data = data
        return self.d_size_map[data]


