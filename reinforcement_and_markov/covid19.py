import numpy as np
import random


def fill_shifted_diagonal(P, prob_list):
    n = P.shape[0]
    for i in range(n-2):
        P[i, i+1] = prob_list[i]
    return P


def fill_right_column(P, prob_list):
    n = P.shape[0]
    for i in range(n-2):
        P[i, -1] = 1 - prob_list[i]
    return P


def create_prob_list(n, complex=False):
    list_len = n-1
    prob_list = []
    for i in range(list_len):
        prob_list.append(1 - (i+1) /(list_len-1) * .49)
    if complex == False:
        return prob_list
    else:
        prob_list = []
        for i in range(list_len):
            print(i," (1 - (i+1) /(list_len-1) * .49) * i%2: ", (1 - (i+1) /(list_len-1) * .98) * np.mod(i,2))
            prob_list.append((1 - (i+1) /(list_len-1) * .98) * np.mod(i,2))
    return prob_list


def create_covid_mpd(weeks, complex = False, healthy_reward = 100):
    n = weeks+1
    dim = (n,n)
    P_stay = np.zeros(dim)
    prob_list = [1] * (n-2)
    P_stay = fill_shifted_diagonal(P_stay, prob_list)
    P_stay = fill_right_column(P_stay, prob_list)
    P_stay[-1,-1] = 1
    P_stay[-2,-2] = 1

    P_go = np.zeros(dim)
    prob_list = create_prob_list(n, complex=complex)
    P_go = fill_shifted_diagonal(P_go, prob_list)
    P_go = fill_right_column(P_go, prob_list)
    P_go[-1,-1] = 1
    P_go[-2,-2] = 1
    P = np.vstack(([P_stay], [P_go]))

    R = np.zeros((n, 2))
    R[:,0] = -2
    R[:,1] = 5
    R[-2,1] = 100
    R[-2,0] = 100
    print(R)
    temp  = P[0,0,0]
    print("temp:   ",type(temp))
    return P, R


def evaluate_reward(P,R,policy, give_states=False):
    start_state = 0
    current_state = start_state
    rewards = []
    states = []
    num_states = len(P[0,0,:])
    desired_iterations = num_states-1
    indexes = list(range(num_states))
    for i in range(desired_iterations):
        states.append(current_state)
        next_step = policy[current_state]
        rewards.append(R[current_state, next_step])
        next_probabilities = P[next_step, current_state, :]
        next_index = random.choices(indexes, weights=next_probabilities, k=1)
        current_state = next_index[0]
    end_reward = sum(rewards)
    if give_states == True:
        return end_reward, rewards, states
    return end_reward


def evaluate_reward_average(P,R,policy, iterations=100):
    list_of_end_rewards = []
    for i in range(iterations):
        list_of_end_rewards.append(evaluate_reward(P,R,policy))
    average_end_reward = sum(list_of_end_rewards) / len(list_of_end_rewards)
    return average_end_reward


P = np.zeros((7,7))
print(fill_shifted_diagonal(P, create_prob_list(7)))
print(fill_right_column(P, create_prob_list(7)))
P, R = create_covid_mpd(6, complex=True)
print(P)
print(R)
print(evaluate_reward(P,R,[0,0,0,0,0,1,0]))
