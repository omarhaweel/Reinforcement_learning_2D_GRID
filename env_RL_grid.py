import numpy as np
import random

alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 300

GRID_ROWS = 4
GRID_COLS = 4
ACTIONS = ['↑', '↓', '←', '→']

# Initialize Q-table [row][col][action]
Q = np.zeros((GRID_ROWS, GRID_COLS, len(ACTIONS)))


start = (0, 1)  # Starting position
goal = (3, 3) # Goal position

def is_terminal(state):
    return state == goal

def get_next_state(state, action):
    x, y = state  
    
    if action == '↑':
        y = min(GRID_ROWS - 1, y + 1)
    elif action == '↓':
        y = max(0, y - 1)
    elif action == '←':
        x = max(0, x - 1)
    elif action == '→':
        x = min(GRID_COLS - 1, x + 1)
    
    return (x, y)

def get_reward(state):
    return 10 if state == goal else -1

for episode in range(episodes):
    # always start from the start state
    state = start

    while not is_terminal(state):
        print(f"Episode {episode + 1}, State: {state}")
        x, y = state

        if random.uniform(0, 1) < epsilon:
            action_idx = random.randint(0, 3)
        else:
            action_idx = np.argmax(Q[y, x])

        # take action
        action = ACTIONS[action_idx]
        # Get next state based on action
        next_state = get_next_state(state, action)
        nx, ny = next_state

        # set reward for next state
        reward = get_reward(next_state)

        # Update Q-value    
        Q[y, x, action_idx] += alpha * (
            reward + gamma * np.max(Q[ny, nx]) - Q[y, x, action_idx]
        )

        state = next_state
  

def display(Q, goal):
    grid = [['' for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

    for y in range(GRID_ROWS):
        for x in range(GRID_COLS):
            best_action = np.argmax(Q[y, x])
            if (x, y) == goal:
                grid[y][x] = 'O'
            elif (x,y) == start:
                grid[y][x] = '*' + ACTIONS[best_action]
            else:
            
                grid[y][x] = ACTIONS[best_action]

    for row in reversed(grid):
        print(row)

display(Q, goal)