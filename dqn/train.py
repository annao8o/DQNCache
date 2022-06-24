import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from dqn.dqn import DQN
from dqn.actions import get_action_space, get_action
from dqn.replay_memory import ReplayMemory
from dqn.environment_wrapper import EnvironmentWrapper
import utils.plotting as plotting
from enum import Enum, auto

class DQNTrainer:
    def __init__(self, env, params, model_path):
        self.params = params
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.replay_memory = ReplayMemory(self.params.memory_capacity)
        self.env = env
        self.environment = EnvironmentWrapper(env, self.params.skip_steps)
        # print(self.environment.reset())
        # print(self.environment.reset().shape)
        self.current_q_net = DQN(input_shape=self.environment.reset().shape, num_of_actions=self.env.num_actions)
        self.current_q_net.to(self.device)
        self.target_q_net = DQN(input_shape=self.environment.reset().shape, num_of_actions=self.env.num_actions)
        self.target_q_net.to(self.device)
        self.optimizer = optim.RMSprop(self.current_q_net.parameters(),
                                       lr=self.params.lr)

    def run(self):
        print("CUDA:", torch.cuda.is_available())
        smooth = 3
        DQN_rewards = []

        stats = plotting.EpisodeStats(episode_lengths=np.zeros(int(self.params.num_of_steps)),
                             episode_rewards=np.zeros(int(self.params.num_of_steps)))
        state = torch.tensor(self.environment.reset(), device=self.device, dtype=torch.float32)

        for step in range(int(self.params.num_of_steps)):
            print(f'Episode {step} start!')
            self.env.episode = step

            # Reset the environment
            state = torch.tensor(self.environment.reset(),
                                 device=self.device,
                                 dtype=torch.float32)

            for t in range(self.env.end_T - 1):
                self.env.load_curr_request(t)
                # RL choose action based on (t-1) observation
                q_value = self.current_q_net(torch.stack([state]))
                # Take action. The agent choose action based on observation
                action_index, action = get_action(self.env.actions, q_value,
                                                  train=True,
                                                  step=step,
                                                  params=self.params,
                                                  device=self.device)

                # Process the action and get the new observation
                reward, next_state, done = self.environment.step(action)
                # print(t, reward, done)
                next_state = torch.tensor(next_state,
                                          device=self.device,
                                          dtype=torch.float32)
                self.replay_memory.add(state, action_index, reward, next_state, done)

                if done:
                    break  # add

                if len(self.replay_memory.memory) > self.params.batch_size:  # batch_size 만큼 replay memory에 data 쌓이면 network training
                    loss = self._update_current_q_net()
                    print('Update: {}. Loss: {}'.format(t, loss))

                if step % self.params.target_update_freq == 0:
                    self._update_target_q_net()

                state = next_state

                stats.episode_rewards[step] += reward
                stats.episode_lengths[step] = t

            print(f'Episode {step}/{self.params.num_of_steps}\nReward: {stats.episode_rewards[step]}')
            DQN_rewards.append(self.env.total_reward)
            # hits.append(self.total_hits / self.env.total_requests)

        plotting.plot_episode_stats(stats, smooth)
        plotting.record_rewards(DQN_rewards)
        torch.save(self.target_q_net.state_dict(), self.model_path)


    def _update_current_q_net(self):
        batch = self.replay_memory.sample(self.params.batch_size)
        # self.replay_memory.clear()
        states, actions, rewards, next_states, dones = batch

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.stack(actions).view(-1, 1)
        rewards = torch.tensor(rewards, device=self.device)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)

        q_values = self.current_q_net(states).gather(1, actions)
        next_q_values = self.target_q_net(next_states).max(1)[0]

        expected_q_values = rewards + self.params.discount_factor * next_q_values * (1 - dones)
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


    def _update_target_q_net(self):
        self.target_q_net.load_state_dict(self.current_q_net.state_dict())


    # def __dataloader(self) -> DataLoader:
    #     """Initialize the Replay Buffer dataset used for retrieving experiences."""
    #     dataset = RLDataset(self.buffer, self.hparams.episode_length)
    #     dataloader = DataLoader(
    #         dataset=dataset,
    #         batch_size=self.hparams.batch_size,
    #     )
    #     return dataloader
    #
    # def train_dataloader(self) -> DataLoader:
    #     """Get train loader."""
    #     return self.__dataloader()