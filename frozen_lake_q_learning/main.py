import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

seed = 0  # Random number generator seed
gamma = 0.95  # Discount factor
num_iters = 300  # Number of iterations
alpha   = 0.9  # Learing rate
epsilon = 0.9  # Epsilon in epsilion gready algorithm
random.seed(seed)  # Set the random seed
np.random.seed(seed)

# Now set up the environment
env = gym.make('FrozenLake-v1',
                    desc=None,
                    map_name="4x4",
                    is_slippery=False,
                    render_mode="human")
env.metadata['render_fps'] = 90

observation, info = env.reset()
num_states = env.observation_space.n
num_actions = env.action_space.n
map = env.unwrapped.desc
def e_greedy(env, Q, s, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()

    else:
        return np.argmax(Q[s,:])


Q  = np.zeros((num_states, num_actions))
V  = np.zeros((num_iters + 1, num_states))
pi = np.zeros((num_iters + 1, num_states))

for k in range(1, num_iters + 1):
    # Decay epsilon (exploration rate)
    if k >256:
        epsilon=0.3
    # Reset environment
    state, done = env.reset(), False
    state=state[0]
    print(f"\rIteration: {k}/{num_iters}", end="")

    while not done:
        # Select an action for a given state and acts in env based on selected action
        action = e_greedy(env, Q, state, epsilon)
        next_state, reward, done, _, _ = env.step(action)

        # Q-update:
        y = reward + gamma * np.max(Q[next_state,:])
        Q[state, action] = Q[state, action] + alpha * (y - Q[state, action])

        # Move to the next state
        state = next_state
    # Record max value and max action for visualization purpose only
    for s in range(num_states):
        V[k,s]  = np.max(Q[s,:])
        pi[k,s] = np.argmax(Q[s,:])

    
# Value function subplot
V_grid = V[k,:].reshape(4,4)
im = plt.imshow(V_grid, cmap='viridis')
# Add policy arrows
for i in range(4):
    for j in range(4):

        s = i*4 + j
        action = int(pi[k,s])
        if action == 0:  # left
            dx, dy = -0.2, 0
        elif action == 1:  # down
            dx, dy = 0, 0.2
        elif action == 2:  # right
            dx, dy = 0.2, 0
        else:  # up
            dx, dy = 0, -0.2
        if map[i][j]==b'H':
            plt.text(j, i, 'H', ha='center', va='center', color='white')
        elif map[i][j]==b'G':
            plt.text(j, i, 'G', ha='center', va='center', color='white')
        else:
            plt.arrow(j, i, dx, dy, head_width=0.1, head_length=0.1, fc='white', ec='white')

plt.title(f'k={k}')
plt.xticks(range(4))
plt.yticks(range(4))
plt.show()
print("Done")