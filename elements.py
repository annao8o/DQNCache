import numpy as np
from numpy.linalg import norm
import networkx as nx
from itertools import combinations
import random
from datetime import timedelta
from collections import deque
import matplotlib.pyplot as plt

# def make_process_event(svr, data):
#     wait_delay[svr.id] += data.queueing_time
#     trans_delay +=


class Data:
    def __init__(self, id, size, lifetime):
        self.id = id
        self.size = size
        self.lifetime = lifetime
        self.connected_svr = None

    def set_svr(self, svr):
        self.connected_svr = svr

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return self.idx == other.idx
        else:
            return False

    def process_request(self):
        return


# class RequestedData(Data):
#     def __init__(self, t, data, source):
#         super().__init__(data.id, data.size, data.lifetime)
#         self.request_time = t
#         self.source = source
#         self.queueing_time = None
#         self.status = None
#
#     def update_status(self, curr_t):
#         self.queueing_time = curr_t - self.request_time
#
#
#     def set_queueing(self, T):
#         self.queueing_time = T
#
#     def __eq__(self, other):
#         if type(self) == type(other):
#             return super().__eq__(other) and self.request_time == other.request_time
#         else:
#             return False


class Server:
    def __init__(self, id, cache_size, num_data, env):
        self.id = id
        self.capacity = cache_size
        self.env = env
        self.num_data = num_data
        self.caching_map = np.zeros(num_data, dtype=np.bool_)
        self.storage_usage = 0
        self.queue = deque()
        self.requests_for_data = np.zeros(num_data, dtype=np.int_)
        self.lambda_ = np.zeros(self.num_data, dtype=np.float_)

    def clear(self):
        self.requests_for_data = np.zeros(self.num_data, dtype=np.int_)
        self.lambda_ = np.zeros(self.num_data, dtype=np.float_)

    def request(self, data):
        self.requests_for_data[data.id] += 1

    def get_lambda(self, total_req):
        self.lambda_ = self.requests_for_data / total_req
        return self.lambda_

    # def calc_lambda_(self, arrival_rate, popularity):
    #     self.lambda_ = self.caching_map * arrival_rate * popularity
    #     return self.lambda_


    def isContain(self, data):
        if type(data) == int:
            data_idx = data
        elif isinstance(data, Data):
            data_idx = data.id
        else:
            raise Exception("Func isContain: type %s cannot be handled" % type(data))

        return self.caching_map[data_idx]


    # def calc_storage_usage(self):
    #     self.storage_usage = np.sum(self.env.d_size_map * self.caching_map)
    #     return self.storage_usage
    #
    # def get_usable_storage(self):
    #     return self.capacity - self.calc_storage_usage()
    #
    # def check_cache_avail(self, data):
    #     if data.size < self.get_usable_storage():
    #         return True
    #     else:
    #         print("No enough capacity")
    #         return False

    # def data_store(self, cache_item, freshness):
    #     if type(cache_item) == np.ndarray:
    #         self.caching_map = cache_item
    #     elif type(cache_item) == int:
    #         self.caching_map[cache_item] = True
    #     else:
    #         raise Exception("wrong type error: cache data must integer or array")
    #
    #     self.freshness = self.caching_map * freshness     # data의 lifetime 설정



    # def update_status(self):
    #     # freshness 1씩 감소
    #     for f in range(len(self.freshness)):
    #         if self.freshness[f] == -1:
    #             self.freshness[f] = -1
    #         else:
    #             self.freshness[f] -= 1


        # for data in self.queue:
        #     data.update_status(curr_t)
        #     if data.status == 'END':
        #         self.queue.remove(data)

    # def calc_wait_q(self):
    #     q_data = 0  # the sum of data in server queue
    #     for data in self.queue:
    #         q_data += data.size
    #     return q_data

    # ##queue 안에 있는 request 처리
    # def proc_request(self):
    #     data = self.pop_request()
    #     if data != -1:
    #         self.processing_event = data
    #
    #
    #     self.queue[0].status = 'PROCESS'
    #     # for req in self.queue:
    #     #     req.
    #
    #
    # def pop_request(self):
    #     if self.queue:
    #         return self.queue.popleft()
    #     else:
    #         return -1
    #
    # def processing_end(self):
    #     self.processing_event = None
    #     kwargs = self.pop_request()
    #     if kwargs != -1:
    #         self.processing_event = kwargs['data']
    #         # make_process_event(self, data)


class Zipf:
    def __init__(self):
        self.pdf = None
        self.cdf = None

    def set_env(self, expn, num_contents):
        temp = np.power(np.arange(1, num_contents + 1), -expn)
        zeta = np.r_[0.0, np.cumsum(temp)]
        # zeta = np.r_[0.0, temp]
        self.pdf = [x / zeta[-1] for x in temp]
        self.cdf = [x / zeta[-1] for x in zeta]

    def get_sample(self, size=None):
        if size is None:
            f = random.random()
        else:
            f = np.random.random(size)
        return np.searchsorted(self.cdf, f) - 1


''''
class Controller:   # controller
    def __init__(self, num_svr, num_data, arrival_rate):
        self.num_svr = num_svr
        self.num_data = num_data
        self.arrival_rate = arrival_rate

        self.svr_lst = list()
        self.data_lst = list()

        self.caching_map = np.zeros((num_svr, num_data), dtype=np.bool_)
        self.graph = None
        self.d_size_map = np.zeros(num_data, dtype=np.int_)
        self.rtt_map = np.zeros((num_svr, num_svr), dtype=np.float_)
        self.usable_storage = np.zeros(self.num_svr)
        self.algo_lst = list()


    def create_env(self, data_lst, cache_size):
        self.svr_lst = [Server(i, cache_size, self.num_data) for i in range(self.num_svr)]

        if type(data_lst) == list:
            self.data_lst = data_lst
        elif type(data_lst) == dict:
            self.data_lst = list(data_lst.values())
        else:
            raise Exception("wrong type error: %s is not acceptable for data list" % type(data_lst))

        for data in self.data_lst:
            self.d_size_map[data.id] = data.size

        g = nx.Graph()
        for i in self.svr_lst:
            g.add_node(i)
            self.usable_storage[i] = i.get_usable_storage()

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


    # def add_request(self, request):
    #     hit_lst = list()
    #     delay_lst = list()
    #
    #     for algo in self.algo_lst:
    #         hit, delay = algo.check_cache(request.svr, request.id)
    #         hit_lst.append(hit)
    #         delay_lst.append(delay)
    #     # print("server_id {}: {}".format(self.id, self.queue))
    #     return hit_lst, delay_lst

    # def set_svr_cache(self, svr_idx, cache_item):
    #     if type(cache_item) == np.ndarray:
    #         self.caching_map[svr_idx, :] = cache_item
    #     elif type(cache_item) == list:
    #         if len(cache_item) == self.env['num data']:
    #             self.caching_map[svr_idx, :] = np.array(cache_item)
    #         else:
    #             raise Exception("array dimension unmatch: cache data must interger or array")
    # 
    #     elif type(cache_item) == int:
    #         self.caching_map[svr_idx, cache_item] = 1
    #     else:
    #         raise Exception("wrong type error: cache data must interger or array")
    
'''

