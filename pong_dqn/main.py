import gymnasium as gym
import numpy as np
from model import PongModel

# Create the Pong environment
env = gym.make("Pong-v4", render_mode="human")
replay_memory = np.zeros((1_000_000, 210, 160, 3))
Q = PongModel()
# Game loop
while True:
    # Reset the environment
    #observation is a 210x160x3 array representing the screen
    # info is {'lives': 0, 'episode_frame_number': 0, 'frame_number': 0}
    observation, info = env.reset()
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        # Render the game
        env.render()
        
        # Get action from keyboard (0: NOOP, 2: UP, 3: DOWN)
        action = int(input("Enter action (0: NOOP, 2: UP, 3: DOWN): "))
        
        # Take action in environment
        observation, reward, terminated, truncated, info = env.step(action)
    
    if input("Play again? (y/n): ").lower() != 'y':
        break

env.close()

