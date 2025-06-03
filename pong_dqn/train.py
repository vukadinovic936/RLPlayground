import gymnasium as gym
import numpy as np
from model import PongModel
import matplotlib.pyplot as plt
import cv2
import torch

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
env = gym.make("Pong-v4")
replay_memory = np.zeros((1_000_000, 210, 160, 3))
# initialize Q with random weights 
Q = PongModel()
#hyperparams
M=1000
T=100
epsilon = 0.05
for episode in range(M):
    #observation is a 210x160x3 array representing the screen
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
    print("new episode")
    for t in range(T):
        # info is {'lives': 0, 'episode_frame_number': 0, 'frame_number': 0}
        if  terminated or truncated:
            break
        else:
            if np.random.rand() < epsilon:
                # take a random action selecting from 0,1 and 2
                action = np.random.randint(0, 3)
            else:
                # Convert preprocessed sequence to tensor and get Q values
                state = torch.stack(preprocessed_sequence[-4:]).unsqueeze(0)
                action = Q(state).squeeze().argmax().item()
            print(action)
            # Take action in environment
            # this is how actions are defined in pong environment 0,2,3
            env_action = action+1
            if env_action == 1:
                env_action = 0
            
            observation, reward, terminated, truncated, info = env.step(env_action)
            sequence.append(observation)
            preprocessed_observation  = preprocess_observation(observation) 
            preprocessed_sequence.append(preprocessed_observation)

            plt.imshow(preprocessed_sequence[-1])
            plt.show()

    env.close()

