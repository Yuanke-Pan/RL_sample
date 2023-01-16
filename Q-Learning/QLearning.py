import torch
import torch.optim as optim
import torch.functional as F
import torch.nn as nn
import math
import numpy as np
import random

from torchvision.transforms import Compose, ToTensor, Normalize

from model import RLNet
from ReplayBuffer import ReplayBuffer

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD



class DQN(object):
    def __init__(self, model, target_model, memory, cfg) -> None:
        self.n_actions = cfg["n_actions"]
        self.device = torch.device(cfg['device'])
        self.gamma = cfg['gamma']
        self.sample_count = 0
        self.epsilon = cfg['epsilon_start']
        self.epsilon_start = cfg['epsilon_start']
        self.epsilon_end = cfg['epsilon_end']
        self.epsilon_decay = cfg['epsilon_decay']
        self.batchsize = cfg['batch_size']

        self.policy_net = model.to(self.device)
        self.target_net = target_model.to(self.device)
        self.target_net.eval()

        #把policy网络的参数复制到target网络
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg['lr'])
        self.memory = memory

        self.state_transforms = Compose([ToTensor()])

        self.target_count = 0
        self.target_flash_rate = 1000

    def sample_action(self, state):
        self.sample_count += 1
        
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay)
        
        state = np.array(state) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.n_actions)
        
        return action
    
    @torch.no_grad()
    def predict_action(self, state):
        
        state = torch.tensor(state).to(self.device)
        q_values = self.policy_net(state)
        action = q_values.max(1)[1].item()
        return action
    
    def update(self):
        if len(self.memory) < self.batchsize:
            return
        self.target_count += 1
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batchsize)
        state_batch = torch.from_numpy(state_batch).float().to(self.device)
        state_batch = state_batch / 255.0
        action_batch = torch.from_numpy(action_batch).long().to(self.device)
        reward_batch = torch.from_numpy(reward_batch).float().to(self.device)
        next_state_batch = torch.from_numpy(next_state_batch).float().to(self.device)
        next_state_batch = next_state_batch / 255.0
        done_batch = torch.from_numpy(done_batch).float().to(self.device)
        
        with torch.no_grad():
            _, max_next_action = self.policy_net(next_state_batch).max(1)
            next_q_values = self.target_net(next_state_batch).gather(1, max_next_action.unsqueeze(1)).squeeze()
            expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
        
        loss = nn.MSELoss()(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()
        if self.target_count % self.target_flash_rate == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())