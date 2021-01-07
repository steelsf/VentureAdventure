import numpy as np
import pandas as pd
import mdp_hiive_steel as mdp
import hiive.mdptoolbox as mdptoolbox
from hiive.mdptoolbox import example
#import mdptoolbox
import hiive.mdptoolbox.example as example
import examples_copied as example
import grid_example
import tictactoe
import matplotlib.pyplot as plt
import covid19


def run_stat_to_df(solver, algorithm, problem, size, discount, epsilon):
    run_stats = solver.run_stats
    temp_df = pd.DataFrame(run_stats)
    temp_df['algorithm'] = algorithm
    temp_df['problem'] = problem
    temp_df['size'] = size
    temp_df['discount'] = discount
    temp_df['epsilon'] = epsilon
    return temp_df


def rewards_from_policy(problem, P, R, policy, grid_params=None, iterations=100):
    if problem == 'Stir Crazy':
        reward = covid19.evaluate_reward_average(P,R,policy)
    else:
        reward = grid_example.evaluate_reward(P,R,policy,grid_params)
    return reward


def key_metrics(P, R, solver, algorithm, problem, size, discount, epsilon, grid_params=None):
    policy = solver.policy
    reward = rewards_from_policy(problem,  P, R, policy, grid_params=grid_params)
    run_stats = solver.run_stats
    last_row_stats = run_stats[-1]
    temp_dict = {'Time':last_row_stats['Time'], 'Iteration':last_row_stats['Iteration']\
    , 'reward':reward}
    return temp_dict

#problem,size,
def run_learner(algorithm,problem,size,P,R,discount,grid_params=None,epsilon=1\
, max_iter=1000, n_iter=1000000, initial_value=0, verbose=False,output_location=None,save_df=False):
    ''' Inputs: algo, prob, size, prefix, output_loc, P, R, discount, (parameters for tuning)
        Outputs: By iteration (error, mean V, time, iteration)
                 For each final outcome (policy, rewards (simulated and averaged))
                 Attached to each (algo, prob, size, gamma, epsilon (QL only))
        Charts:  Iter vs time, iter vs error, iter vs mean V
    '''
    print("running learner for ",algorithm," ",problem," ",size)
    if algorithm == 'VI':
        solver=mdp.ValueIteration(P,R,discount, max_iter=max_iter)
    elif algorithm == 'PI':
        solver=mdp.PolicyIteration(P,R,discount, max_iter=max_iter)
    elif algorithm == 'QL':
        solver=mdp.QLearning(P,R,discount,n_iter=n_iter,epsilon=epsilon)
    else:
        print("Incorrect keyword given for algorithm to use. Returning None")
        return None
    if verbose == True:
        solver.setVerbose()
    else:
        solver.setSilent()
    solver.run()
    run_stats_df = run_stat_to_df(solver, algorithm, problem, size, discount, epsilon)
    policy = solver.policy
    key_metrics_dict = key_metrics(P, R, solver, algorithm, problem, size, discount\
    ,epsilon, grid_params=grid_params)
    full_title = algorithm + " " + problem + " " + size
    policy_df = pd.DataFrame(policy,columns = [full_title])
    master_dict = {'run_stats_df':run_stats_df, 'key_metrics_dict':key_metrics_dict, 'policy':policy_df}

    if save_df == True:
        save_learner_single_run(algorithm,problem,size, master_dict, output_location, discount=discount\
        , epsilon=epsilon)
    return master_dict


def save_learner_single_run(algorithm,problem,size, master_dict, output_location, discount=.9, epsilon=1):
    temp_title = algorithm + " " + problem + " " + size + " G=" + str(discount)
    if algorithm == "QL":
        temp_title = algorithm + " " + problem + " " + size + " G=" + str(discount) + " E=" + str(epsilon)
    run_stats_df = master_dict['run_stats_df']
    policy_df = master_dict['policy']
    key_metrics_df = pd.DataFrame(master_dict['key_metrics_dict'], index=[temp_title])
    run_stats_df.to_csv(output_location + "run_stats/" + temp_title + " _run_stats_df.csv")
    policy_df.to_csv(output_location + "policy/" + temp_title + " _policy_df.csv")
    key_metrics_df.to_csv(output_location + "single_key_metric/" +  temp_title \
    + " _key_metrics_df.csv", index=temp_title)
    return


def save_learner_multiple_run(names_list, master_dict, output_location):
    num = len(names_list)
    mean_v_df = pd.DateFrame({'Iteration'})
    for i in num:
        run_stats_df = master_dict['run_stats_df']
        policy_df = master_dict['policy_df']
        key_metrics = pd.DataFrame(master_dict['key_metrics_dict'])
    return


def run_learner_diff_gamma(algorithm,P,R,discount,epsilon=0.01, max_iter=1000, n_iter=10000\
, initial_value=0, verbose=False):
    '''Return a chart and df for diff gammas on same prob and size
    '''
    return

def run_learner_diff_gamma_epsilon(algorithm,P,R,discount,epsilon=0.01, max_iter=1000, n_iter=10000\
, initial_value=0, verbose=False):
    '''Return a 3 charts and 1df for diff epsilon (and charts for each gamma) on same prob and size
    '''
    #for QLearning only
    return



def run_three_learners(P,R,discount,epsilon=0.01, max_iter=1000, n_iter=10000, initial_value=0\
,summary_only=False):
    algo_list = ['VI', 'PI', 'QL']
    vi_master_dict = run_learner('VI',P,R,discount,epsilon=epsilon, max_iter=max_iter)
    pi_master_dict = run_learner('PI',P,R,discount, max_iter=max_iter)
    ql_master_dict = run_learner('QL',P,R,discount, n_iter=n_iter)
    vi_metrics_dict = vi_master_dict['key_metrics_dict']
    pi_metrics_dict = pi_master_dict['key_metrics_dict']
    ql_metrics_dict = ql_master_dict['key_metrics_dict']
    time_list = [vi_metrics_dict['Time'], pi_metrics_dict['Time'], ql_metrics_dict['Time']]
    iter_list = [vi_metrics_dict['Iteration'], pi_metrics_dict['Iteration']\
    , ql_metrics_dict['Iteration']]
    rewards_list = [vi_metrics_dict['Reward'], pi_metrics_dict['Reward']\
    , ql_metrics_dict['Reward']]
    key_metrics_df = pd.DataFrame({'Algorithm':algo_list, 'Time':time_list, 'Iteration'\
    :iter_list, 'Reward':rewards_list})
    if summary_only == True:
        return key_metrics_df
    else:
        return vi_master_dict, pi_master_dict, ql_master_dict


def chart_curve(x_array, y_array, title, output_location, labels_list=0\
, use_f1=False, xlabel=None, ylabel=None):
    plt.figure(figsize=(6.4*.5,4.8*.5))
    if y_array.ndim == 1:
        num_lines = 1
    else:
        num_lines = y_array.shape[0]
    if num_lines == 1:
        plt.plot(x_array, y_array)
    else:
        for i in range(num_lines):
            plt.plot(x_array, y_array[i], label = labels_list[i])
    plt.title(title, fontsize=10)
    plt.grid(False)
    plt.xlabel(xlabel,fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    #plt.figure(figsize=(6.4*.5,4.8*.5))
    plt.tight_layout()
    if labels_list == 0:
        labels_list = list(range(num_lines))
    else:
        plt.legend(fontsize=10)
    plt.savefig(output_location+title+'.png')
    plt.clf()
    return


def loop_run_learner(algorithm_list, problem_list, size_list, gamma_list, ol, grid_dictionary=None\
,stir_num_dict=None, save_df=False):
    for p in problem_list:
        for s in size_list:
            if p == "C19 Grid":
                P, R = grid_example.grid_world_example(param_dict=grid_dictionary[s])
            else:
                P, R = covid19.create_covid_mpd(stir_num_dict[s], complex=False)
            for a in algorithm_list:
                for g in gamma_list:
                    print("for ",p," ",s," ",a," ",g)
                    temp = run_learner(a,p,s,P,R,g,grid_params=grid_dictionary[s]\
                    ,output_location=ol,save_df=True)
    return


def loop_run_learner_ql(algorithm_list, problem_list, size_list, gamma_list, epsilon_list, ol\
, grid_dictionary=None,stir_num_dict=None, save_df=False):
    for p in problem_list:
        for s in size_list:
            if p == "C19 Grid":
                P, R = grid_example.grid_world_example(param_dict=grid_dictionary[s])
            else:
                P, R = covid19.create_covid_mpd(stir_num_dict[s], complex=False)
            for a in algorithm_list:
                for g in gamma_list:
                    for e in epsilon_list:
                        print("for ",p," ",s," ",a," ",g," ",e)
                        temp = run_learner(a,p,s,P,R,g,grid_params=grid_dictionary[s]\
                        ,output_location=ol,save_df=True, epsilon=e)
    return


def main():
    project_folder = '/Users/vwy957/Documents/ML/markov/'
    output_location = project_folder + 'outputs/'
    ol = project_folder + 'outputs/'

    algorithm_list = ['PI', 'VI']
    problem_list = [ 'Stir Crazy','C19 Grid',]
    size_list = ['Small', 'Medium', 'Large']
    gamma_list = [.99, .95, .90, .80]

    #epsilon_list = ['E=0.8 ','E=0.9 ', 'E=0.99 ']

    # P, R = covid19.create_covid_mpd(31, complex=False)
    grid_small = grid_example.get_small_problem_params()
    grid_medium = grid_example.get_medium_problem_params()
    grid_large = grid_example.get_large_problem_params()
    grid_dictionary = {'Small':grid_small, 'Medium':grid_medium, 'Large':grid_large}
    stir_num_dict = {'Small':7, 'Medium':52, 'Large':365}

    # loop_run_learner(algorithm_list, problem_list, size_list, gamma_list, ol\
    # , grid_dictionary=grid_dictionary, stir_num_dict=stir_num_dict, save_df=True)


    algorithm_list = ['QL']
    problem_list = [ 'Stir Crazy','C19 Grid',]
    size_list = ['Small', 'Medium'] #, 'Large']
    gamma_list = [.99, .90, .80, .70]
    epsilon_list = [.99, .90, .80, .70]
    loop_run_learner_ql(algorithm_list, problem_list, size_list, gamma_list, epsilon_list, ol\
    , grid_dictionary=grid_dictionary, stir_num_dict=stir_num_dict, save_df=True)

if __name__ == "__main__":
    main()




### Tasks ###
'''
Remaining tasks:
Bar charts for reward
Print outs for optimal policy
QL charts (second round)

'''
