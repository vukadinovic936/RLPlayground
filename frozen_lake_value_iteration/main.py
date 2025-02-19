import random
import numpy as np 
import matplotlib.pyplot as plt

seed=0
gamma = 0.95  # Discount factor
num_iters = 10  # Number of iterations

random.seed(seed)  # Set the random seed to ensure results can be reproduced
np.random.seed(seed)

def next_state(s,a):
    """
    s: current state
    a: action
    return: next state
    """
    next_state = None
    if a==0:  # left
        next_state = s-1 if s % 4 != 0 else s  # Don't move left if already at left edge
    elif a==1:  # down
        next_state = s+4 if s < 12 else s  # Don't move down if already at bottom edge
    elif a==2:  # right
        next_state = s+1 if (s+1) % 4 != 0 else s  # Don't move right if already at right edge
    elif a==3:  # up
        next_state = s-4 if s >= 4 else s  # Don't move up if already at top edge
    return next_state


action_space = range(4) # 0-left, 1-down, 2-right, 3-up 
state_space = range(16) # cur_row*n_cols + cur_col
reward = np.zeros((len(state_space),len(action_space)))
reward[14,2]=1
#four holes and goal.
terminal_states = [5,7,11,12,15]

# set Q to some random values
Q = np.zeros((num_iters+1,len(state_space),len(action_space)))
V = np.zeros((num_iters+1,len(state_space)))
pi = np.zeros((num_iters+1,len(state_space)))
for k in range(1,num_iters+1):
    for s in state_space:
        #terminal states
        if s in terminal_states:
            continue

        for a in action_space:
            r=reward[s,a]
            Q[k,s,a] += r + gamma * V[k-1,next_state(s,a)]
        
        V[k,s] = np.max(Q[k,s,:])
        pi[k,s] = np.argmax(Q[k,s,:])

# Create a figure with 10 subplots (2 rows, 5 columns)
plt.figure(figsize=(20, 8))
for k in range(1,num_iters+1):
    # Create subplot for each iteration
    plt.subplot(2, 5, k)
    
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
            if s in terminal_states[:-1]:
                plt.text(j, i, 'H', ha='center', va='center', color='white')
            elif s==terminal_states[-1]:
                plt.text(j, i, 'G', ha='center', va='center', color='white')
            else:
                plt.arrow(j, i, dx, dy, head_width=0.1, head_length=0.1, fc='white', ec='white')
    
    plt.title(f'k={k}')
    plt.xticks(range(4))
    plt.yticks(range(4))

# Add a colorbar that spans all subplots
#plt.colorbar(im, ax=plt.gcf().axes, label='Value')
plt.suptitle('Value Function Evolution Over Iterations', fontsize=16)
plt.tight_layout()
plt.show()

## add transition probability