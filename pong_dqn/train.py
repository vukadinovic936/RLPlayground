import gymnasium as gym
import numpy as np
from model import PongModel
import matplotlib.pyplot as plt
import cv2
import torch
import random 
import copy
from collections import deque

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
env = gym.make("Pong-v4")#,render_mode="human")
# should contain tuples of (state, action, reward, next_state, done)
replay_memory = deque(maxlen=100_000)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# initialize Q with random weights 
Q = PongModel().to(device)


#hyperparams
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(Q.parameters(), lr=0.0001)
M=1000
T=10_000
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 0.999
gamma = 0.99
total_steps = 0

losses_through_episodes = []
for episode in range(M):
    episode_losses = []

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
    episode_reward = 0
    print(f"Episode {episode}")

    for t in range(T):
        total_steps += 1
        # Calculate current epsilon
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** total_steps))
        state = torch.stack(preprocessed_sequence[-4:]).unsqueeze(0).to(device)

        if terminated or truncated:
            break
        else:
            if np.random.rand() < epsilon:
                # take a random action selecting from 0,1 and 2
                action = np.random.randint(0, 3)
            else:
                # Convert preprocessed sequence to tensor and get Q values
                with torch.no_grad():
                    action = Q(state).squeeze().argmax().item()
            
            # Take action in environment
            # Get action from keyboard (0: NOOP, 2: UP, 3: DOWN)
            env_action = action+1
            if env_action == 1:
                env_action = 0

            total_reward = 0
            for _ in range(4): 
                observation, reward, terminated, truncated, info = env.step(env_action)
                total_reward += reward
                sequence.append(observation)
                preprocessed_observation = preprocess_observation(observation)
                preprocessed_sequence.append(preprocessed_observation)
                if terminated or truncated:
                    break

            next_state = torch.stack(preprocessed_sequence[-4:])
            replay_memory.append(
                (state.squeeze(0).clone().detach(), action, total_reward,
                next_state.squeeze(0).clone().detach(), terminated or truncated)
            )
            episode_reward += total_reward
            # Now we do gradient update
            # Only sample if we have enough transitions in replay memory
            if len(replay_memory) >= 1000:
                # sample a batch of size 32 from replay memory
                batch = random.sample(replay_memory, 32)
                states, actions, rewards, next_states, dones = zip(*batch)
                # Convert to tensors
                states      = torch.stack(states).to(device)                 # [32, 4, 84, 84]
                actions     = torch.tensor(actions, dtype=torch.long).to(device)  # [32]
                rewards     = torch.tensor(rewards, dtype=torch.float32).to(device)  # [32]
                next_states = torch.stack(next_states).to(device)            # [32, 4, 84, 84]
                dones       = torch.tensor(dones, dtype=torch.bool).to(device)      # [32]     


                # Compute next step Q values
                with torch.no_grad():
                    q_next = Q(next_states)                       # [32, num_actions]
                    max_q_next = q_next.max(dim=1).values         # [32]
                
                # Compute targets (loss fcn from algorithm)
                y = rewards + (~dones) * gamma * max_q_next       # [32]

                # get losses
                x=states
                preds = Q(x)[torch.arange(32), actions]
                loss = loss_fn(preds, y)
                
                optimizer.zero_grad()
                loss.backward()
                episode_losses.append(loss.item())
                optimizer.step()
                
    
    print(f"Episode {episode} finished with reward {episode_reward}")
    print(f"Episode loss: {np.mean(episode_losses)}")
    losses_through_episodes.append(np.mean(episode_losses))
    # Save model checkpoint
    if episode % 100 == 0:  # Save every 100 episodes
        checkpoint = {
            'episode': episode,
            'model_state_dict': Q.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': np.mean(episode_losses),
        }
        torch.save(checkpoint, f'dqn_checkpoint_episode_{episode}.pth')
        
    # Plot losses
    plt.figure(figsize=(10,5))
    plt.plot(losses_through_episodes)
    plt.title('Training Loss Through Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.savefig(f'loss_plot_episode.png')
    plt.close()

env.close()

