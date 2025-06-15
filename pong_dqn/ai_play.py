import gymnasium as gym
import numpy as np
from model import PongModel
import matplotlib.pyplot as plt
import cv2
import torch
import random 
import copy
from collections import deque
from tqdm import tqdm
def preprocess_observation(observation):
    # resize from 210x160x3 to 110x84 grayscale
    observation = cv2.resize(observation, (110, 84))
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    # cut out 84x84
    observation = torch.tensor(observation[:,18:102], dtype=torch.float32)
    #normalize
    observation = observation / 255.0
    return observation

# Create the Pong environment
env = gym.make("Pong-v4",render_mode="human")
# should contain tuples of (state, action, reward, next_state, done)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# initialize Q with random weights 
Q = PongModel().to(device)
Q.load_state_dict(torch.load('dqn_checkpoint_episode_900.pth',weights_only=False,map_location=device)['model_state_dict'])

total_reward = 0
#for i in tqdm(range(100)):
episode_reward=0
observation, info = env.reset()
sequence = []
preprocessed_sequence = []
# initialize the sequence for this episode, add 4 so that Q can run
for _ in range(4):
    sequence.append(observation)
    preprocessed_observation  = preprocess_observation(observation) 
    preprocessed_sequence.append(preprocessed_observation)

terminated = False
truncated = False

while not terminated and not truncated:
    state = torch.stack(preprocessed_sequence[-4:]).unsqueeze(0).to(device)

    # Convert preprocessed sequence to tensor and get Q values
    with torch.no_grad():
        action = Q(state).squeeze().argmax().item()
    
    # Take action in environment
    # Get action from keyboard (0: NOOP, 2: UP, 3: DOWN)
    env_action = action+1
    if env_action == 1:
        env_action = 0

    for _ in range(4): 
        observation, reward, terminated, truncated, info = env.step(env_action)
        episode_reward += reward
        sequence.append(observation)
        preprocessed_observation = preprocess_observation(observation)
        preprocessed_sequence.append(preprocessed_observation)
        if terminated or truncated:
            break 
print(episode_reward)
total_reward += episode_reward


env.close()
print(total_reward/100)
