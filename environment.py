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

        self.observation_length = 3
        self.num_svr = num_svr
        self.cache_size = cache_size
        self.arrival_rate = arrival_rate

        self.end_T = end_T
        self.curr_t = 0
        self.update_period = update_period
        self.requests = requests

        self.svr_lst = list()

        self.curr_req = list()
        self.caching_actions = list()
        self.num_actions = 0
        self.actions =[[] for _ in range(self.num_svr)]

        self.freshness = np.full((self.num_svr, self.num_data), -1, dtype=np.int_)
        self.cache_mat = np.zeros((self.num_svr, self.num_data), dtype=np.bool_)
        self.lambda_ = np.zeros((self.num_svr, self.num_data), dtype=np.float_)
        self.wait_queue = np.zeros((self.num_svr, self.num_data), dtype=np.float_)
        self.req_cnt = 0
        self.costs = list()

        self.graph = None
        self.rtt_map = np.zeros((self.num_svr, self.num_svr), dtype=np.float_)
        # self.usable_storage = np.zeros(self.num_svr)

        self.create_env()

    def create_env(self):
        print("create env...")
        self.svr_lst = [Server(i, self.cache_size, self.num_data, self) for i in range(self.num_svr)]
        self.def_action_space()

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

    def def_action_space(self):
        print("action space definition starts")
        tmp = list(map(list, itertools.product([0, 1], repeat=self.num_data)))
        for item in tmp:
            if item.count(1) == (self.cache_size / data_size):       # statistics.mean(self.d_size_map)
                self.caching_actions.append(item)
        action_idxs = [i for i in range(len(self.caching_actions))]
        self.actions = list(itertools.permutations(action_idxs,3))
        self.num_actions = len(self.actions)
        print(f'Action space size is {self.num_actions}')


    def load_curr_request(self, t):   # set requests of current time t
        self.curr_req.clear()
        self.curr_t = t
        for r in self.requests:
            if r[0] == t:
                self.curr_req.append(r)
        # print(f'time {t}, request length {len(self.curr_req)}')


    def add_request(self, req):    # set request of each edge server at current time t
        t, svr, data = req
        self.svr_lst[svr].add_request(t, data)


    def isContain(self, data):
        if type(data) == int:
            data_idx = data
        elif isinstance(data, Data):
            data_idx = data.id
        else:
            raise Exception("Func isContain: type %s cannot be handled" % type(data))

        cached_svr = np.where(self.cache_mat[:, data_idx])[0].tolist()
        if cached_svr:
            return self.svr_lst[cached_svr[0]]      # rtt가 제일 짧은 서버로 바꾸기
        else:
            return None


    def reset(self):
        self.curr_t = 0
        self.total_requests = 0
        self.total_reward = 0
        self.curr_req = list()
        self.costs = list()

        self.lambda_ = np.zeros((self.num_svr, self.num_data), dtype=np.float_)
        self.cache_mat = np.zeros((self.num_svr, self.num_data), dtype=np.bool)
        self.freshness = np.full((self.num_svr, self.num_data), -1, dtype=np.int_)
        # self.wait_queue = np.zeros((self.num_svr, self.num_data), dtype=np.float_)

        for s in self.svr_lst:
            s.clear()

        return self.get_observation()


    def step(self, action):     # [(action_idx-svr1, action_idx-svr2,...), ...]
        done = False
        # print(self.curr_t, action)
        for s, a in enumerate(action):
            self.addToCache(s, np.array(self.caching_actions[a]))

        # Current request 만큼 loop 돌려서 state update 하기
        for req in self.curr_req:
            # request cnt (arrival rate) update
            self.add_request(req)
            self.total_requests += 1

        self.update_state()

        reward = self.reward_func()     # saved retrieval delay
        # print("Reward: ", reward)
        self.total_reward += reward

        observation_ = self.get_observation()

        if (self.curr_t == self.end_T-1):
            done = True

        return reward, observation_, done


        '''
        # Caching the data in server according the chosen action
        if action is not None:
            sel_svr = action[0]
            sel_data = action[1]
            self.addToCache(sel_svr, sel_data)

            # self.update_state(action)

        # Get new state
        # new_state = self.get_observation()
        self.update_state()
        state_ = self.get_observation()
        reward = self.reward_func(action[0])
        self.total_reward += reward
        done = self.hasDone()

        return state_, reward, done
        '''

    def reward_func(self):
        past_cost = 0
        if self.costs:
            past_cost = self.costs[-1]
        curr_cost = self.calc_total_time(self.calc_trans_time(), self.calc_wait_time())
        # print("Curr_reward: ", curr_cost)
        self.costs.append(curr_cost)

        return curr_cost - past_cost


    def addToCache(self, svr, caching_d):
        # print(caching_d)
        self.cache_mat[svr] = caching_d
        self.svr_lst[svr].data_store(caching_d, self.data_lifetime * caching_d)


    def calc_popularity(self):
        popularity = np.zeros(self.num_data, dtype=np.float_)

        for s in self.svr_lst:
            self.total_requests += s.total_request
            popularity = np.add(popularity, s.requests_for_data)
        popularity = popularity / self.total_requests

        return popularity


    def update_state(self):
        # update cache_mat, freshness, wait_queue
        popularity = self.calc_popularity()

        for s in self.svr_lst:
            caching_map, freshness, request_rate = s.report(self.arrival_rate, popularity)
            self.cache_mat[s.id] = caching_map
            self.freshness[s.id] = freshness
            # self.wait_queue[s.id][0] = queue_status
            self.lambda_[s.id] = request_rate

        # print(self.curr_t, self.lambda_)
            # s.update_status()


    def get_observation(self):
        return np.stack((self.lambda_.copy(), self.freshness.copy(), self.cache_mat.copy()), axis=0)

        # return dict(lambda_=self.lambda_.copy(),
        #             freshness=self.freshness.copy(),
        #             cache_status=self.cache_mat.copy(),
        #             queue_status=self.wait_queue.copy()
        #             )

    # def hasDone(self):
    #     return ((self.curr_t % self.update_period) == 0)


    # def check_capacity(self, svr):
    #     sum_data = 0
    #     data_idx = np.where(self.cache_mat[svr] == 1)
    #     if np.any(data_idx):
    #         for data in data_idx[0]:
    #             sum_data += self.get_size(data)
    #
    #     if sum_data > self.cache_size:
    #         return False
    #     else:
    #         return True



    # def reward_func(self, svr):
    #     # Reward: (average transmission delay + average waiting delay) + Penalty term
    #     penalty_param = 1.0
    #     reward = self.calc_total_time(self.calc_trans_time(), self.calc_wait_time())
    #     # d_old = self.calc_total_time(self.calc_trans_time(old), self.calc_wait_time(old))
    #     # d_new = self.calc_total_time(self.calc_trans_time(new), self.calc_wait_time(new))
    #
    #     # if d_old < d_new:
    #     #     reward = 0
    #     # else:
    #     #     reward = 1
    #
    #     exceed_flag = False
    #     if not self.check_capacity(svr):
    #         exceed_flag = True
    #
    #     reward -= exceed_flag * penalty_param
    #     return reward


    def calc_trans_time(self):
        avg_trans = np.zeros(self.num_svr, dtype=np.float_)
        for m in range(self.num_svr):
            d_m = 0
            d_nm = 0
            d_device = 0
            for i in range(self.num_data):
                d_size = self.get_size(i)
                if self.cache_mat[m][i] is True and self.freshness[m][i] > 0:    #data i is cached in source server m
                    d_m = d_size/env_params['backhaul']
                elif list(np.nonzero(self.cache_mat[:, i])[0]): #data i is cached in neighbor server n
                    neighbor_svr = list(np.nonzero(self.cache_mat[:, i])[0])
                    if self.freshness[m][neighbor_svr[0]] > 0:
                        rtt = self.rtt_map[m][neighbor_svr[0]]
                        d_nm = d_size * rtt
                else:
                    R_m = env_params['bandwidth']*math.log2(1+((env_params['power']*env_params['channel gain']) / (env_params['noise power']*env_params['bandwidth'])))
                    d_device = d_size/R_m

                avg_trans[m] += self.lambda_[m][i] * (d_m + d_nm + d_device)
            # print(self.lambda_[m], avg_trans[m])
        # print("Calculate transmission time", avg_trans)
        return avg_trans


    def calc_wait_time(self):
        avg_wait = np.zeros(self.num_svr, dtype=np.float_)
        for m in range(self.num_svr):
            e1 = 0
            e2 = 0
            for i in range(self.num_data):
                service_t = self.get_size(i) / env_params['backhaul']
                e2 += pow(service_t, 2) * self.lambda_[m][i]
                e1 += service_t * self.lambda_[m][i]
            avg_wait[m] = e2 / (2 * (1 - e1))
        # print("Calculate waiting time", avg_wait)
        return avg_wait

    def calc_total_time(self, t, w):
        tmp = np.add(t, w)
        return np.sum(tmp)

    def get_size(self, data):
        if type(data) == Data:
            data = data.id
        elif type(data) == int:
            data = data
        return self.d_size_map[data]


    # def data_store(self):
    #     caching_result = self.cache_mat.copy()
    #     for svr in self.svr_lst:
    #         svr.data_store(caching_result[svr.id], self.data_lifetime)
