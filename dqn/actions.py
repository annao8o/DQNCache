import torch
from utils.config import *



def get_action_space(ACTIONS):
    return len(ACTIONS)


def get_action(ACTIONS, q_value, train=False, step=None, params=None, device=None):
    if train:
        epsilon = params.epsilon_final + (params.epsilon_start - params.epsilon_final) * \
            math.exp(-1 * step / params.epsilon_step)
        if random.random() <= epsilon:
            action_index = random.randrange(get_action_space(ACTIONS))
            action = ACTIONS[action_index]
            return torch.tensor([action_index], device=device)[0], action
    action_index = q_value.max(1)[1]
    action = ACTIONS[action_index[0]]
    return action_index[0], action