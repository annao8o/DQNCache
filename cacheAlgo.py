import numpy as np
from utils.config import *
from itertools import combinations

class CacheAlgo:
    def __init__(self, name, env, replace_algo):
        self.algo_name = name
        self.env = env
        self.caching_map = np.zeros((env.num_svr, env.num_data), dtype=np.bool_)
        self.freshness = np.full((env.num_svr, env.num_data), -1, dtype=np.int_)
        self.activation_cnt = 0
        self.threshold = np.zeros(env.num_svr, dtype=np.float_)
        self.replace_algo = replace_algo

    def clear(self):
        self.caching_map = np.zeros((self.env.num_svr, self.env.num_data), dtype=np.bool_)
        self.freshness = np.full((self.env.num_svr, self.env.num_data), -1, dtype=np.int_)

    def proactive_caching(self):
        if self.algo_name == "proposed":
            self.proposed_caching()
        elif self.algo_name == "optimal":
            self.optimal_caching()
        elif self.algo_name == "random":
            self.random_caching()
        elif self.algo_name == "MPC":
            self.MPC_caching()
        print(f'{self.algo_name}>>>>caching!\n{self.caching_map}')

    def update_caching(self):
        if self.replace_algo == "proposed":
            self.replace_proposed()
        elif self.replace_algo == "LRU":
            self.replace_LRU()


    def trans_direct(self, target, data):
        size = self.env.d_size_map[data]
        R_m = env_params['bandwidth']*math.log2(1+((env_params['power']*env_params['channel gain']) / (env_params['noise power']*env_params['bandwidth'])))
        d2b_time = size/R_m     #device to broker
        d_m = size / env_params['backhaul']
        rtt = self.env.rtt_map[self.env.data_lst[data].connected_svr][target]
        # print(d2b_time + rtt/2 + d_m)
        return d2b_time + rtt/2 + d_m

    def trans_edge(self, svr, target, data):
        # min_rtt = float('inf')
        # if type(svr) == int:
        min_rtt = self.env.rtt_map[svr][target]
        # elif type(svr) == np.ndarray:
        #     for s in svr:
        #         rtt = self.env.rtt_map[s][target]
        #         if min_rtt > rtt:
        #             min_rtt = rtt
        size = self.env.d_size_map[data]
        d_m = size / env_params['backhaul']
        # print((min_rtt/2) + d_m)
        return (min_rtt/2) + d_m

    def proposed_caching(self):
        # print("proposed caching")
        avail_data = [i for i in range(self.env.num_data)]
        for svr in range(self.env.num_svr):
            avail_storage = self.env.cache_size
            while avail_storage > 0:
                max_data = None
                max_gain = 0
                # print(avail_data)
                for data in avail_data:
                    gain = self.calc_gain(svr, data)
                    # print(data, gain)
                    if gain > max_gain:
                        max_gain = gain
                        max_data = data
                # print(max_data)
                if max_data==None:
                    max_data = avail_data[random.randint(0, len(avail_data)-1)]
                self.data_store(svr, max_data)
                # print(f"caching data {max_data} in server {svr}")
                avail_storage -= self.env.d_size_map[max_data]
                avail_data.remove(max_data)

                '''
                for data in range(self.env.num_data):
                if


                if len(self.check_edge_cache(data))!=0:
                    gain_list.append((data, 0))
                else:
                    gain = self.calc_gain(svr, data)
                    gain_list.append((data, gain))
            sorted_gain = sorted(gain_list, key=lambda tup: tup[1], reverse=True)
            print(sorted_gain)

            idx = 0
            while avail_storage > 0:
                max_data = sorted_gain[idx][0]
                self.data_store(svr, max_data)
                avail_storage -= self.env.d_size_map[max_data]
                idx += 1
            '''
        '''
        for data in range(self.env.num_data):
            max_svr = -1
            max_gain = 0
            for svr in range(self.env.num_svr):
                if avail_storage[svr] >= self.env.d_size_map[data]:
                    gain = self.calc_gain(svr, data)
                    # print("gain: ", gain)
                    if (gain > 0) and (gain > max_gain):
                        max_svr = svr
                        max_gain = gain
            if max_gain > 0:
                self.data_store(max_svr, data, self.env.data_lifetime)
                avail_storage[max_svr] -= self.env.d_size_map[data]
        '''

    def calc_gain(self, svr, data):
        gain = 0
        for target_svr in range(self.env.num_svr):
            req_rate = self.env.lambda_i[data]
            # print(f"data {data}, req_rate {req_rate}")
            # p_i = self.calc_freshProb(req_rate)
            gain += req_rate*(self.trans_direct(target_svr, data) - self.trans_edge(svr, target_svr, data))
        return gain

    def optimal_caching(self):
        # def calc_delay():
        #     delay = 0.0
        #     for svr in range(self.env.num_svr):
        #         for data in range(self.env.num_data):
        #             req_rate = self.env.lambda_i[data]
        #             cached_svr = self.check_edge_cache(data)
        #             if len(cached_svr)!=0:
        #                 d = req_rate*self.trans_edge(cached_svr, svr, data)
        #             else:
        #                 d = req_rate*self.trans_direct(svr, data)
        #             delay += d
        #     return delay

        full_status = [False for _ in range(self.env.num_svr)]
        def isfull():
            status = True
            for s in full_status:
                status *= s
            return status
        gain_lst = list()
        avail_size = [self.env.cache_size for _ in range(self.env.num_svr)]
        for svr in range(self.env.num_svr):
            for data in range(self.env.num_data):
                gain = self.calc_gain(svr, data)
                gain_lst.append((svr, data, gain))  #gain_lst = (server id, data id, gain value)
        sorted_gain = sorted(gain_lst, key=lambda tup: tup[2], reverse=True)
        while isfull()==False:
            svr, data, gain = sorted_gain.pop(0)
            # print(svr,data,gain)
            if len(np.where(self.caching_map[:, data] == True)[0]):
                continue
            if avail_size[svr] == 0:
                full_status[svr] = True
            else:
                self.data_store(svr, data)
                avail_size[svr] -= self.env.d_size_map[data]


        '''
        gain_lst = []
        for svr in range(self.env.num_svr):
            for data in range(self.env.num_data):
                gain = self.calc_gain(svr, data)
                gain_lst.append((svr, data, gain))
        sorted_gain = sorted(gain_lst, key=lambda tup: tup[2], reverse=True)
        # print(sorted_gain)
        while isfull()==False:
            svr, data, gain = sorted_gain.pop(0)
            # print(svr,data,gain)
            if len(np.where(self.caching_map[:, data] == True)[0]):
                continue
            if avail_size[svr] == 0:
                full_status[svr] = True
            else:
                self.data_store(svr, data, self.env.data_lifetime)
                avail_size[svr] -= self.env.d_size_map[data]
        '''
        '''
        def calc_delay():
            delay = 0.0
            for svr in range(self.env.num_svr):
                for data in range(self.env.num_data):
                    req_rate = self.env.lambda_i[data]
                    cached_svr = self.check_edge_cache(data)
                    if len(cached_svr)!=0:
                        d = req_rate*self.trans_edge(cached_svr, svr, data)
                    else:
                        d = req_rate*self.trans_direct(svr, data)
                    delay += d
            return delay

        data = [i for i in range(self.env.num_data)]
        comb = list(combinations(data, self.env.cache_size))
        all_comb = list(combinations(comb, self.env.num_svr))
        min_delay = float('inf')
        min_comb = None

        for idx, comb in enumerate(all_comb):
            # print(f"{idx}/{len(all_comb)}")
            for s, item in enumerate(comb):
                self.data_store(s, list(item))
            delay = calc_delay()
            if delay < min_delay:
                min_delay = delay
                min_comb = comb

        self.cache_clear()
        # delay 작은 combination으로 data store
        for s, d in enumerate(min_comb):
            print(f"select data {d} in server {s}")
            self.data_store(s, list(d))
        '''

    def check_edge_cache(self, data):
        return np.where(self.caching_map[:, data]==True)[0]

    def cache_clear(self):
        self.caching_map = np.zeros((self.env.num_svr, self.env.num_data), dtype=np.bool_)


    def random_caching(self):
        # print("random caching")
        for svr in range(self.env.num_svr):
            avail_storage = self.env.cache_size
            while avail_storage > 0:
                avail_data = np.where(avail_storage >= self.env.d_size_map)[0]
                avail_data = np.delete(avail_data, np.where(self.caching_map[svr]==True))
                data_idx = avail_data[random.randint(0, len(avail_data) - 1)]
                if not self.caching_map[svr][data_idx]:
                    self.data_store(svr, int(data_idx))
                    avail_storage -= self.env.d_size_map[data_idx]


    def MPC_caching(self):
        # print("MPC caching")
        for svr in range(self.env.num_svr):
            avail_storage = self.env.cache_size
            while avail_storage > 0:
                avail_data = np.where(avail_storage >= self.env.d_size_map)[0]
                avail_data = np.delete(avail_data, np.where(self.caching_map[svr]==True))
                max_lambda = 0
                popular_data = None
                # print(len(avail_data))
                for d in avail_data:
                    # print(self.caching_map[svr])
                    # print(self.env.lambda_[svr])
                    # print(self.caching_map[svr][d], self.env.lambda_[svr][d])
                    if (not self.caching_map[svr][d]) and (max_lambda < self.env.lambda_[svr][d]):
                        # print("aaa")
                        max_lambda = self.env.lambda_[svr][d]
                        popular_data = d
                    # print(svr, popular_data)
                if popular_data is not None:
                    self.data_store(svr, int(popular_data))
                    # print("data store")
                    avail_storage -= self.env.d_size_map[popular_data]


    def data_store(self, svr, cache_item):
        if type(cache_item) == np.ndarray:
            self.caching_map[svr] = cache_item
        elif type(cache_item) == int:
            self.caching_map[svr][cache_item] = True
        elif type(cache_item) == list:
            for d in cache_item:
                self.caching_map[svr][d] = True
        else:
            raise Exception("wrong type error: cache data must integer or array")
        self.freshness[svr] = self.caching_map[svr] * self.env.data_lifetime     # data의 lifetime 설정


    def update_cache(self, alpha):
        for svr in range(self.env.num_svr):
            lifetime_zero = np.where(self.freshness[svr]==0)[0]  # find the data which has freshness 0
            avg_lambda = np.mean(self.caching_map[svr] * self.env.lambda_i)
            threshold_t = (1 - alpha) * self.threshold[svr] + alpha * avg_lambda
            # print(lifetime_zero)
            if len(lifetime_zero) > 0:
                for data in lifetime_zero:
                    # print(f'Algorithm: {self.algo_name} ---> The arrival rate of data {data} is {self.env.lambda_i[data]}, threshold is {threshold_t}')
                    if self.env.lambda_i[data] >= threshold_t:
                        # print(f'Update data {data} in {svr}')
                        self.freshness[svr][data] = self.env.data_lifetime[data]
                        # self.activation_cnt += 1
            self.threshold[svr] = threshold_t


    def request(self, req): #req=(t, svr, data obj)
        req_svr = req[1]
        req_data = req[2].id
        cached_svr = np.where(self.caching_map[:, req_data]==True)[0]
        hit = 0
        delay = 0
        if (len(cached_svr) != 0) and (self.check_freshness(cached_svr[0], req_data)):  #if data is cached in edge
            hit = 1
            delay = self.trans_edge(cached_svr[0], req_svr, req_data)   #고치기
        else:
            hit = 0
            delay = self.trans_direct(req_svr, req_data)
            self.activation_cnt += 1
            # print(delay)
        return hit, delay


    def update_fresh(self):
        self.freshness -= 1
        # print("Data Lifetime ----> ", self.env.data_lifetime)
        # print("Freshness ----> ", self.freshness)


    def check_freshness(self, svr, data):
        freshness = self.freshness[svr][data]
        if freshness > 0:
            return True
        else:
            return False


    '''
    if self.caching_map[es_idx, f]:
        print(self.algo_name, ">>>>", f, "is in the ", es_idx, "es")
        hit = 1
        delay = 0
    else:   # check the neighboring ES
        status = list()
        neighbor_idx = list()
        for n in range(len(self.rtt_map[es_idx])):
            if self.rtt_map[es_idx, n] == 0:
                continue
            if self.caching_map[n, f]:
                status.append(self.rtt_map[es_idx, n])
                neighbor_idx.append(n)
        if status:
            hit = 1
            shortest_idx = status.index(min(status))
            delay = min(status) * self.env['rtt']
            print(self.algo_name, ">>>>", f, "is in the neighboring ", neighbor_idx[shortest_idx], "es")
        else:
            print(self.algo_name, ">>>>", f, "is not cached")
            hit = 0
            delay = self.env['cloud rtt']

    return hit, delay
    '''




