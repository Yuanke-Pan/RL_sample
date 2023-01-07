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
    def __init__(self, model, memory, cfg) -> None:
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
        self.target_net = model.to(self.device)
        #把policy网络的参数复制到target网络
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=cfg['lr'])
        self.memory = memory

        self.state_transforms = Compose([ToTensor(), Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    def convert_state(self, state):
        if isinstance(state, tuple):
            state = [s for s in state]
        else:
            state = [state]
        result = []
        for s in state:
            result.append(self.state_transforms(s).unsqueeze(0))
            
        result = torch.concatenate(result)
        result = result.to(self.device)
        
        return result

    def sample_action(self, state):
        self.sample_count += 1
        
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay)
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = self.convert_state(state)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.n_actions)
        
        return action
    
    @torch.no_grad()
    def predict_action(self, state):
        
        state = self.convert_state(state)
        q_values = self.policy_net(state)
        action = q_values.max(1)[1].item()
        return action
    
    def update(self):
        if len(self.memory) < self.batchsize:
            return
        
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batchsize)
        state_batch = self.convert_state(state_batch)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)  
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = self.convert_state(next_state_batch)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()

        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()