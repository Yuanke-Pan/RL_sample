from collections import deque
import random

class ReplayBuffer(object):
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
    
    def push(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batchsize, sequential = False):
        
        if batchsize > len(self.buffer):
            batchsize = len(self.buffer)
        
        if sequential:
            rand = random.randint(0, len(self.buffer) - batchsize)
            batch = [self.buffer[i] for i in range(rand, rand + batchsize)]
        else:
            batch = random.sample(self.buffer, batchsize)
        
        return zip(*batch)
    
    def clear(self):
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)