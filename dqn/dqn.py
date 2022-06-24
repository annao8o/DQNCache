import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, num_of_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_of_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # print('input size: ', x.shape)
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
'''
class DQN(nn.Module):
    def __init__(self, input_shape, num_of_actions):
        super().__init__()

        self.conv1 = nn.Conv2d(input_shape, 16, kernel_size=1, stride=2)
        print('kernel_size:', self.conv1.weight.shape)
        # self.conv1.in_channels = 1
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=1, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=1, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(32)


        self.linear1 = nn.Linear(32*7*7, 256)
        self.linear2 = nn.Linear(256, num_of_actions)

    def forward(self, x):
        print('input size: ', x.shape)
        conv1_out = F.relu(self.batch_norm1(self.conv1(x)))
        conv2_out = F.relu(self.batch_norm2(self.conv2(conv1_out)))
        conv3_out = F.relu(self.batch_norm3(self.conv3(conv2_out)))

        flattened = torch.flatten(conv3_out, start_dim=1)
        linear1_out = self.linear1(flattened)
        q_value = self.linear2(linear1_out)

        return q_value
'''

if __name__ == '__main__':
    x = torch.rand(1, 1, 84, 84)
    dqn = DQN(input_shape=1, num_of_actions=4)
    a = dqn(x)
    m = a.max(1)[1]
    print(a)
    print(m)
