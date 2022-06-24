import os, sys
import numpy as np
from datetime import timedelta
from elements import Zipf, Data


def make_request_events(num_svr, arrival_rate, end_t, zipf, data_lst):
    request_events = list()
    # t = timedelta(0)
    # t_interval = timedelta(seconds=interval)
    # end_t = timedelta(end_t)
    # print(end_t)
    t = 0

    while t < end_t:
        req_num = np.random.poisson(arrival_rate, size=num_svr)
        for svr_idx in range(num_svr):
            if req_num[svr_idx] != 0:
                for _ in range(req_num[svr_idx]):
                    request_events.append((t, svr_idx, data_lst[zipf.get_sample()]))

        t += 1
    request_events.sort(key=lambda x: x[0])
    return request_events


def make_data(num_svr, num_data, size, freshness_range):
    data_list = list()
    #size_list = np.random.uniform(size_range, size=num_data)
    freshness_list = np.random.randint(freshness_range[0], freshness_range[1], size=num_data)
    connected_svr = np.random.randint(low=0, high=num_svr, size=num_data)

    for d_idx in range(num_data):
        data = Data(d_idx, size, freshness_list[d_idx])
        data.set_svr(connected_svr[d_idx])
        data_list.append(data)

    return data_list

