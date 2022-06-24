import random
import math

gamma = 0.99    #discount factor
lr = 0.01
eta = 0.5

num_data = 1000
num_server = 4
cache_size = 150     #MB
data_size = 1  #MB
life_time = (10, 100)    #seconds
alpha = 0.125 #EWMA
end_time = 24*60*60
arrival_rate = 10
zipf_param = 0.7
update_period = 6*60*60

dist = 10
env_params = {'bandwidth': 10, 'power': 0.5, 'channel gain': 127+30*math.log(10), 'noise power': 10**-174, 'backhaul': 20}     # MHz, W, ?, dBm/Hz, MHz

model_path = 'save/model/'
model_file = 'model.pt'

file_path = 'save/'
file_name = 'env1(arrival10).bin'