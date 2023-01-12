import gymnasium as gym
import ale_py

import torch
import torch.nn as nn
import random
import numpy as np
import timm

from model import simpleCnn
from ReplayBuffer import ReplayBuffer
from QLearning import DQN

def train(cfg, env, agent):

    print("开始训练！")
    rewards = []
    steps = []
    max_rewards = -100

    for i_ep in range(cfg['train_eps']):
        ep_reward = 0
        ep_step = 0
        state = env.reset()
        for _ in range(cfg['ep_max_steps']):
            ep_step += 1
            action = agent.sample_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push((state, action, reward, next_state, done))
            state = next_state
            agent.update()
            ep_reward += reward
            if done:
                break
        
        if (i_ep + 1) % cfg['target_update'] == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        
        if max_rewards < ep_reward:
            torch.save(agent.policy_net.state_dict(), "bigReward_model.pth")

        steps.append(ep_step)
        rewards.append(ep_reward)

        if (i_ep + 1) % 1 == 0:
            print(f"Epochs: {i_ep + 1} / {cfg['train_eps']}, Reward: {ep_reward:.2f}, Episilon: {agent.epsilon:.3f}")
            torch.save(agent.policy_net.state_dict(), "timestep_model.pth")

    print("train_finish!!!")
    env.close()

    return {'rewards' : rewards}

def test(cfg, env, agent):
    print("test Begin!")
    rewards = []  # 记录所有回合的奖励
    steps = []
    for i_ep in range(cfg['test_eps']):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        for _ in range(cfg['ep_max_steps']):
            ep_step+=1
            action = agent.predict_action(state)  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        print(f"Epoch: {i_ep+1}/{cfg['test_eps']}, rewards: {ep_reward:.2f}")
    print("Finish test!!!")
    env.close()
    return {'rewards':rewards}

import gym
import os
def all_seed(env,seed = 1):
    ''' 万能的seed函数
    '''
    env.seed(seed) # env config
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # config for CPU
    torch.cuda.manual_seed(seed) # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def env_agent_config(cfg):
    env = gym.make(cfg['env_name']) # 创建环境
    if cfg['seed'] !=0:
        all_seed(env,seed=cfg['seed'])
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print(f"状态空间维度：{n_states}，动作空间维度：{n_actions}")
    cfg.update({"n_states":n_states,"n_actions":n_actions}) # 更新n_states和n_actions到cfg参数中
    model = simpleCnn(env.observation_space) # 创建模型
    memory = ReplayBuffer(cfg['memory_capacity'])
    agent = DQN(model, memory, cfg)
    return env,agent

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
def get_args():
    """ 超参数
    """
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='DQN',type=str,help="name of algorithm")
    parser.add_argument('--env_name',default='ALE/Pong-v5',type=str,help="name of environment")
    parser.add_argument('--train_eps',default=200,type=int,help="episodes of training")
    parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing")
    parser.add_argument('--ep_max_steps',default = 1000000,type=int,help="steps per episode, much larger value can simulate infinite steps")
    parser.add_argument('--gamma',default=0.95,type=float,help="discounted factor")
    parser.add_argument('--epsilon_start',default=0.95,type=float,help="initial value of epsilon")
    parser.add_argument('--epsilon_end',default=0.001,type=float,help="final value of epsilon")
    parser.add_argument('--epsilon_decay',default=25000,type=int,help="decay rate of epsilon, the higher value, the slower decay")
    parser.add_argument('--lr',default=0.0001,type=float,help="learning rate")
    parser.add_argument('--memory_capacity',default=100000,type=int,help="memory capacity")
    parser.add_argument('--batch_size',default=64,type=int)
    parser.add_argument('--target_update',default=4,type=int)
    parser.add_argument('--hidden_dim',default=512,type=int)
    parser.add_argument('--device',default='cuda',type=str,help="cpu or cuda") 
    parser.add_argument('--seed',default=10,type=int,help="seed")   
    args = parser.parse_args([])
    args = {**vars(args)}  # 转换成字典类型    
    ## 打印超参数
    print("超参数")
    print(''.join(['=']*80))
    tplt = "{:^20}\t{:^20}\t{:^20}"
    print(tplt.format("Name", "Value", "Type"))
    for k,v in args.items():
        print(tplt.format(k,v,str(type(v))))   
    print(''.join(['=']*80))      
    return args
def smooth(data, weight=0.9):  
    '''用于平滑曲线，类似于Tensorboard中的smooth曲线
    '''
    last = data[0] 
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed

def plot_rewards(rewards,cfg, tag='train'):
    ''' 画图
    '''
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{tag}ing curve on {cfg['device']} of {cfg['algo_name']} for {cfg['env_name']}")
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()
    plt.show()

cfg = get_args() 
# 训练
env, agent = env_agent_config(cfg)
res_dic = train(cfg, env, agent)
 
plot_rewards(res_dic['rewards'], cfg, tag="train")  
# 测试
res_dic = test(cfg, env, agent)
plot_rewards(res_dic['rewards'], cfg, tag="test")  # 画出结果