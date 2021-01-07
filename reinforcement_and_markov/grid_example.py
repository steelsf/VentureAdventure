from hiive.mdptoolbox import mdp
import hiive.mdptoolbox as mdptoolbox
from hiive.mdptoolbox import example
import mdptoolbox
import hiive.mdptoolbox.example as example
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


def generate_coordinates(n,grid_size):
    random.seed(123)
    nrow = grid_size[0]
    ncol = grid_size[1]
    rows = []
    for i in range(n):
        rows.append(random.randint(0,nrow-1))
    cols = []
    for i in range(n):
        cols.append(random.randint(0,ncol-1))
    coor = zip(rows,cols)
    return list(coor)


def get_small_problem_params():
    param_dict = {
    'grid_size':(8, 8),
    'black_cells':[(1,1)],
    'covid_cells':[(7,7), (2,4), (3,3), (6,0)],
    'white_cell_reward':-0.02,
    'green_cells':[(0,3)], #, (0,4)
    'red_cell_loc':(1,3),
    'green_cell_reward':1.0,
    'red_cell_reward':-1.0,
    'action_lrfb_prob':(.1, .1, .8, 0.),
    'start_loc':(7, 0),
    'covid_reward':-1,
    'pink_cell_reward':-0.25,
    'give_visual':False,
    'chart_title':'R Small'
    }
    return param_dict


def get_medium_problem_params():
    param_dict = {
    'grid_size':(20, 20),
    'black_cells':[(1,1)],
    'covid_cells':generate_coordinates(40,(20,20)),
    'white_cell_reward':-0.02,
    'green_cells':[(0,19)], #, (0,4)
    'red_cell_loc':(1,3),
    'green_cell_reward':1.0,
    'red_cell_reward':-1.0,
    'action_lrfb_prob':(.1, .1, .8, 0.),
    'start_loc':(19, 0),
    'covid_reward':-1,
    'pink_cell_reward':-0.25,
    'give_visual':False,
    'chart_title':'R Medium'
    }
    return param_dict


def get_large_problem_params():
    param_dict = {
    'grid_size':(80, 80),
    'black_cells':[(1,1)],
    'covid_cells':generate_coordinates(160,(80,80)),
    'white_cell_reward':-0.02,
    'green_cells':generate_coordinates(2,(80,80)),
    'red_cell_loc':(1,3),
    'green_cell_reward':1.0,
    'red_cell_reward':-1.0,
    'action_lrfb_prob':(.1, .1, .8, 0.),
    'start_loc':(79, 0),
    'covid_reward':-1,
    'pink_cell_reward':-0.25,
    'give_visual':False,
    'chart_title':'R Large'
    }
    return param_dict


def grid_world_example(param_dict=get_small_problem_params()):
    grid_size=param_dict['grid_size']
    black_cells=param_dict['black_cells']
    covid_cells=param_dict['covid_cells']
    white_cell_reward=param_dict['white_cell_reward']
    green_cells=param_dict['green_cells']
    red_cell_loc=param_dict['red_cell_loc']
    green_cell_reward=param_dict['green_cell_reward']
    red_cell_reward=param_dict['red_cell_reward']
    action_lrfb_prob=param_dict['action_lrfb_prob']
    start_loc=param_dict['start_loc']
    covid_reward=param_dict['covid_reward']
    pink_cell_reward=param_dict['pink_cell_reward']
    give_visual=param_dict['give_visual']
    num_states = grid_size[0] * grid_size[1]
    num_actions = 4
    P = np.zeros((num_actions, num_states, num_states))
    R = np.zeros((num_states, num_actions))

    # helpers
    to_2d = lambda x: np.unravel_index(x, grid_size)
    to_1d = lambda x: np.ravel_multi_index(x, grid_size)

    def hit_wall(cell):
        if cell in black_cells:
            return True
        try: # ...good enough...
            to_1d(cell)
        except ValueError as e:
            return True
        return False

    # make probs for each action
    a_up = [action_lrfb_prob[i] for i in (0, 1, 2, 3)]
    a_down = [action_lrfb_prob[i] for i in (1, 0, 3, 2)]
    a_left = [action_lrfb_prob[i] for i in (2, 3, 1, 0)]
    a_right = [action_lrfb_prob[i] for i in (3, 2, 0, 1)]
    actions = [a_up, a_down, a_left, a_right]
    for i, a in enumerate(actions):
        actions[i] = {'up':a[2], 'down':a[3], 'left':a[0], 'right':a[1]}

    # work in terms of the 2d grid representation

    def update_P_and_R(cell, new_cell, a_index, a_prob):
        if cell in green_cells:
            P[a_index, to_1d(cell), to_1d(cell)] = 1.0
            R[to_1d(cell), a_index] = green_cell_reward

        elif cell == red_cell_loc:
            P[a_index, to_1d(cell), to_1d(cell)] = 1.0
            R[to_1d(cell), a_index] = red_cell_reward

        elif hit_wall(new_cell):  # add prob to current cell
            P[a_index, to_1d(cell), to_1d(cell)] += a_prob
            R[to_1d(cell), a_index] = white_cell_reward

        else:
            P[a_index, to_1d(cell), to_1d(new_cell)] = a_prob
            R[to_1d(cell), a_index] = white_cell_reward

    for a_index, action in enumerate(actions):
        for cell in np.ndindex(grid_size):
            # up
            new_cell = (cell[0]-1, cell[1])
            update_P_and_R(cell, new_cell, a_index, action['up'])

            # down
            new_cell = (cell[0]+1, cell[1])
            update_P_and_R(cell, new_cell, a_index, action['down'])

            # left
            new_cell = (cell[0], cell[1]-1)
            update_P_and_R(cell, new_cell, a_index, action['left'])

            # right
            new_cell = (cell[0], cell[1]+1)
            update_P_and_R(cell, new_cell, a_index, action['right'])

    for cell in covid_cells:
        R[to_1d(cell), :] = covid_reward
    neighbors = lambda x, y : [(x2, y2) for x2 in range(x-1, x+2)\
        for y2 in range(y-1, y+2) if (-1 < x < X and -1 < y < Y and\
        (x != x2 or y != y2) and (0 <= x2 < X) and (0 <= y2 < Y))]
    X, Y = grid_size
    pink_cells = []
    for item in covid_cells:
        x, y = item
        pink_cells.extend(neighbors(x,y))
    new_pink_cells = set(pink_cells)
    pink_cells = list(new_pink_cells)
    for item in pink_cells:
        temp_coor = to_1d(item)
        temp_val = R[temp_coor, 0]
        if temp_val == white_cell_reward:
            R[to_1d(item), :] = pink_cell_reward
    return P, R


def visualize_directions(policy, grid_size=get_small_problem_params()['grid_size']):
    symbol_policy = replace_to_symbol(policy)
    visual = symbol_policy.reshape(grid_size)
    return visual


def replace_to_symbol(array):
    temp_list = list(array)
    new_list = []
    for item in temp_list:
        if item == 0:
            new_list.append("^")
        elif item == 1:
            new_list.append("v")
        elif item == 2:
            new_list.append("<")
        elif item == 3:
            new_list.append(">")
        else:
            new_list.append("*")
    new_array = np.array(new_list)
    return new_array


def visualize_r(R, grid_size=get_small_problem_params()['grid_size']):
    R_grid = R[:,0]
    R_grid = R_grid.reshape(grid_size)
    return R_grid


def visualize_r_chart(R, output_location, title, grid_size=get_small_problem_params()['grid_size']):
    R_grid = visualize_r(R, grid_size=grid_size)
    color_map = plt.imshow(R_grid)
    color_map.set_cmap("Blues_r")
    plt.colorbar()
    plt.title(title, fontsize=10)
    plt.savefig(output_location + title + ".png")
    print("R visualization is save to ", output_location + title + ".png")
    return


def evaluate_reward(P,R,policy, param_dict, give_states=False):
    grid_size = param_dict['grid_size']
    to_2d = lambda x: np.unravel_index(x, grid_size)
    to_1d = lambda x: np.ravel_multi_index(x, grid_size)
    start_coordinates = param_dict['start_loc']
    start_state = to_1d(start_coordinates)
    green_cells=param_dict['green_cells']
    red_cell_loc=param_dict['red_cell_loc']
    red_state = to_1d(red_cell_loc)
    green_states = []
    for i in range(len(green_cells)):
        green_states.append(to_1d(green_cells[i]))
    current_state = start_state
    rewards = []
    states = []
    num_states = len(P[0,0,:])
    desired_iterations = grid_size[0] * grid_size[1]
    indexes = list(range(num_states))
    for i in range(desired_iterations):
        states.append(current_state)
        next_step = policy[current_state]
        rewards.append(R[current_state, next_step])
        next_probabilities = P[next_step, current_state, :]
        next_index = random.choices(indexes, weights=next_probabilities, k=1)
        current_state = next_index[0]
        if current_state in green_states:
            next_step = policy[current_state]
            rewards.append(R[current_state, next_step])
            states.append(current_state)
            break
        if current_state == red_state:
            next_step = policy[current_state]
            rewards.append(R[current_state, next_step])
            states.append(current_state)
            break
    end_reward = sum(rewards)
    if give_states == True:
        return end_reward, rewards, states
    return end_reward


def evaluate_reward_average(P,R,policy,param_dict, iterations=100):
    list_of_end_rewards = []
    for i in range(iterations):
        list_of_end_rewards.append(evaluate_reward(P,R,policy))
    average_end_reward = sum(list_of_end_rewards) / len(list_of_end_rewards)
    return average_end_reward


project_folder = '/Users/vwy957/Documents/ML/markov/'
output_location = project_folder + 'outputs/'
policy = np.array([0,1,2,3])
print(replace_to_symbol(policy))

policy_visual = visualize_directions(policy, grid_size=(2,2))
print(policy_visual)

print(" >>> GRID COVID <<< ")
grid_params = get_large_problem_params()
grid_params = get_medium_problem_params()
#grid_params = get_small_problem_params()
P, R = grid_world_example(param_dict=grid_params)

visualize_r_chart(R, output_location, grid_params['chart_title'], grid_params['grid_size'])
# print(visualize_r(R, grid_params['grid_size']))
# policy = np.repeat([1,2,3,0,1,2,3,0],8)
# print(evaluate_reward(P,R,policy, grid_params, give_states=True))

input_location = '/Users/vwy957/Documents/ML/markov/outputs/'
folder = input_location + "policy/"
file_name = 'PI C19 Grid Small G=0.8 _policy_df.csv'
file_name = 'PI C19 Grid Medium G=0.99 _policy_df.csv'
policy = list(pd.read_csv(folder+file_name).iloc[:,1])
print(policy)
policy_visual = visualize_directions(policy, grid_size=(20,20))
policy_visual_df = pd.DataFrame(policy_visual)
print(policy_visual)
policy_visual_df.to_csv(folder+'medium_grid_solved.csv')
