import os.path
import pickle
import argparse

from utils.config import *
from environment import Environment
from utils.GenRequest import *
from dqn.train import DQNTrainer
from utils.params import Params
from dqn.inference import dqn_inference
from dqn.evaluate import evaluate_dqn


def load_file(path, name):
    with open(os.path.join(path, name), 'rb') as f:
        file = pickle.load(f)
        print("success to load the file: %s" % path+name)
    return file

def save_file(path, name, obj):
    with open(os.path.join(path, name), 'wb') as f:
        pickle.dump(obj, f)
    print("Success to generate and save the file: %s" % path+name)

def run_training(env, model):
    params = Params('utils/dqn.json')
    trainer = DQNTrainer(env, params, model)
    trainer.run()


def run_inference(model):
    score = dqn_inference(model)
    print('Total score: {0:.2f}'.format(score))


def run_evaluation(model):
    score = evaluate_dqn(model)
    print('Average reward after 100 episodes: {0:.2f}'.format(score))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true',
                       help='Train model.')
    group.add_argument('--inference', action='store_true',
                       help='Model inference.')
    group.add_argument('--evaluate', action='store_true',
                       help='Evaluate model on 100 episodes.')

    args = vars(parser.parse_args())

    if os.path.exists(file_path + file_name):
        file = load_file(file_path, file_name)
        data_list = file['data_list']
        requests = file['requests']
        env = file['environment']
        zipf = file['zipf']
    else:
        data_list = make_data(num_data, data_size, life_time)
        zipf = Zipf()
        zipf.set_env(zipf_param, num_data)
        requests = make_request_events(num_server, arrival_rate, end_time, zipf, data_list)
        env = Environment(requests, end_time, update_period, num_server, cache_size, arrival_rate, data_list)
        integrated_file = {'data_list': data_list, 'requests': requests, 'zipf': zipf, 'environment': env}

        save_file(file_path, file_name, integrated_file)

    model = model_path + model_file

    if args['train']:
        run_training(env, model)
    elif args['evaluate']:
        run_evaluation(model)
    else:
        run_inference(model)
