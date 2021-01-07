#imports
import pandas as pd
import numpy as np
import mlrose_hiive as mlrose
import time
import random
import matplotlib.pyplot as plt

print("I am ready to begin")
def get_params_for_grid_search(keyword, max_iters_list=[500], rand_list = [0,11,22]):
    #params to search for tuning
    dict_of_param_dict = {}
    dict_of_param_dict['GA'] = {
    'pop_size':[100,200],#,1000],
    'mutation_prob':[0.5, 0.1, 0.2],
    'max_attempts':[5,10,30],
    'max_iters':max_iters_list,
    'random_state':rand_list
    }
    dict_of_param_dict['RHC'] = {
    'max_attempts':[30,50,100],  #[5,10,20,50]
    'restarts':[5,10,20],  #[0,1,2,5]
    'max_iters':max_iters_list,
    'random_state':rand_list
    }
    dict_of_param_dict['SA'] = {
    'max_attempts':[10,50,100],
    'init_temp':[1.0,10.0,0.5,20,100,1000],
    'decay':[0.99,0.8,0.5],
    'max_iters':max_iters_list,
    'random_state':rand_list
    }
    dict_of_param_dict['MIMIC'] = {
    'pop_size':[100,150],
    'keep_pct':[0.5,0.2],
    'max_attempts':[10],
    'max_iters':[100],
    'random_state':rand_list
    }
    return dict_of_param_dict[keyword]


def get_best_params(keyword, max_iters_list=[2,4,8,16,32,64,128,256,512,1024], rand_list = [0,11,22,33,44,55,66,77,88,99]):
    #storage location for best params for each case
    dict_of_param_dict = {}
    dict_of_param_dict['GA_NN'] = {
    'pop_size':[200],#,1000],
    'mutation_prob':[0.5],
    'max_attempts':[30],
    'max_iters':max_iters_list,
    'random_state':rand_list
    }
    dict_of_param_dict['GA_4P'] = {
    'pop_size':[100,200],#,1000],
    'mutation_prob':[0.5, 0.1, 0.2],
    'max_attempts':[5,10,30],
    'max_iters':max_iters_list,
    'random_state':rand_list
    }
    dict_of_param_dict['GA_FF'] = {
    'pop_size':[100,200],#,1000],
    'mutation_prob':[0.5, 0.1, 0.2],
    'max_attempts':[5,10,30],
    'max_iters':max_iters_list,
    'random_state':rand_list
    }
    dict_of_param_dict['GA_KS'] = {
    'pop_size':[100,200],#,1000],
    'mutation_prob':[0.5, 0.1, 0.2],
    'max_attempts':[5,10,30],
    'max_iters':max_iters_list,
    'random_state':rand_list
    }

    dict_of_param_dict['RHC'] = {
    'max_attempts':[30,50,100],  #[5,10,20,50]
    'restarts':[5,10,20],  #[0,1,2,5]
    'max_iters':max_iters_list,
    'random_state':rand_list
    }
    dict_of_param_dict['SA'] = {
    'max_attempts':[10,50,100],
    'init_temp':[1.0,10.0,0.5,20,100,1000],
    'decay':[0.99,0.8,0.5],
    'max_iters':max_iters_list,
    'random_state':rand_list
    }
    dict_of_param_dict['MIMIC'] = {
    'pop_size':[100,150],
    'keep_pct':[0.5,0.2],
    'max_attempts':[10],
    'max_iters':[100],
    'random_state':rand_list
    }
    return dict_of_param_dict[keyword]


def call_mlrose(algorith_keyword, problem, pop_size=200, mutation_prob=0.1, max_attempts=10, max_iters=np.inf, curve=False\
, random_state=None, schedule=mlrose.GeomDecay(), init_state=None, restarts=0, keep_pct=0.2, fast_mimic=True):
    if curve == True:
        best_state, best_fitness, curve_output = call_mlrose_curve(algorith_keyword, problem, pop_size=pop_size\
        , mutation_prob=mutation_prob, max_attempts=max_attempts, max_iters=max_iters, curve=curve, random_state=random_state\
        , schedule=schedule, init_state=init_state, restarts=restarts, keep_pct=keep_pct, fast_mimic=fast_mimic)
        return best_state, best_fitness, curve_output
    if algorith_keyword == 'RHC':
        best_state, best_fitness, z = mlrose.random_hill_climb(problem, max_attempts=max_attempts, max_iters=max_iters\
        , restarts=restarts, init_state=init_state, curve=False, random_state=random_state)
    elif algorith_keyword == 'GA':
        best_state, best_fitness, z = mlrose.genetic_alg(problem, pop_size=pop_size, mutation_prob=mutation_prob\
        , max_attempts=max_attempts, max_iters=max_iters, curve=curve, random_state=random_state)
    elif algorith_keyword == 'SA':
        best_state, best_fitness, z = mlrose.simulated_annealing(problem, schedule=schedule, max_attempts=max_attempts\
        ,max_iters=max_iters, init_state=init_state, curve=curve, random_state=random_state)
    elif algorith_keyword == 'MIMIC':
        print("problem: ",problem,"\npop_size: ",pop_size,"\n","keep_pct: ",keep_pct)
        print("max_attempts: ",max_attempts,"\nmax_iters: ",max_iters,"\nrandom_state: ",random_state,"\nfast_mimic: ",fast_mimic)
        best_state, best_fitness, z = mlrose.mimic(problem, pop_size=pop_size, keep_pct=keep_pct\
        , max_attempts=max_attempts, max_iters=max_iters, curve=curve, random_state=random_state)
        print("best_fitness: ",best_fitness)
    else:
        print("\n\nIncorrect 'algorithm_keyword'. Please check the input to the 'call_mlrose' function.\n\n")
        best_state, best_fitness, z = 'incorrect key word', 'incorrect key word', 'incorrect key word'
    return best_state, best_fitness


def call_mlrose_curve(algorith_keyword, problem, pop_size=200, mutation_prob=0.1, max_attempts=10, max_iters=np.inf, curve=False\
, random_state=None, schedule=mlrose.GeomDecay(), init_state=None, restarts=0, keep_pct=0.2, fast_mimic=True):
    if algorith_keyword == 'RHC':
        best_state, best_fitness, curve_output = mlrose.random_hill_climb(problem, max_attempts=max_attempts, max_iters=max_iters\
        , restarts=restarts, init_state=init_state, curve=curve, random_state=random_state)
    elif algorith_keyword == 'GA':
        best_state, best_fitness, curve_output = mlrose.genetic_alg(problem, pop_size=pop_size, mutation_prob=mutation_prob\
        , max_attempts=max_attempts, max_iters=max_iters, curve=curve, random_state=random_state)
    elif algorith_keyword == 'SA':
        best_state, best_fitness, curve_output = mlrose.simulated_annealing(problem, schedule=schedule, max_attempts=max_attempts\
        ,max_iters=max_iters, init_state=init_state, curve=curve, random_state=random_state)
    elif algorith_keyword == 'MIMIC':
        print("problem: ",problem,"\npop_size: ",pop_size,"\n","keep_pct: ",keep_pct)
        print("max_attempts: ",max_attempts,"\nmax_iters: ",max_iters,"\nrandom_state: ",random_state,"\nfast_mimic: ",fast_mimic)
        best_state, best_fitness, curve_output = mlrose.mimic(problem, pop_size=pop_size, keep_pct=keep_pct\
        , max_attempts=max_attempts, max_iters=max_iters, curve=curve, random_state=random_state)
        print("best_fitness: ",best_fitness)
    else:
        print("\n\nIncorrect 'algorithm_keyword'. Please check the input to the 'call_mlrose' function.\n\n")
        best_state, best_fitness, curve_output = 'incorrect key word', 'incorrect key word', 'incorrect key word'
    return best_state, best_fitness, curve_output


def fitness_by_iter(keyword, problem, max_iters_list, rand_list, pop_size=200, mutation_prob=0.1, max_attempts=10, curve=False\
, schedule=mlrose.GeomDecay(), init_state=None, restarts=0, keep_pct=0.2, fast_mimic=True):
    avg_fit_list = []
    max_fit_list = []
    min_fit_list = []
    avg_time_list = []
    curve_output_list = []
    for m in max_iters_list:
        temp_fit_list = []
        temp_time_list = []
        temp_curve_list = []
        for r in rand_list:
            if curve == False:
                start_time_fit = time.perf_counter()
                best_state, best_fitness = call_mlrose(keyword, problem, random_state=r, max_iters=m, max_attempts=max_attempts\
                , pop_size=pop_size, mutation_prob=mutation_prob,  curve=curve, schedule=schedule\
                , init_state=init_state, restarts=restarts, keep_pct=keep_pct, fast_mimic=fast_mimic)
                end_time_fit = time.perf_counter()
                time_used = end_time_fit - start_time_fit
                temp_fit_list.append(best_fitness)
                temp_time_list.append(time_used)
            else:
                start_time_fit = time.perf_counter()
                best_state, best_fitness, curve_output = call_mlrose(keyword, problem, random_state=r, max_iters=m\
                , max_attempts=max_attempts, pop_size=pop_size, mutation_prob=mutation_prob,  curve=True, schedule=schedule\
                , init_state=init_state, restarts=restarts, keep_pct=keep_pct, fast_mimic=fast_mimic)
                end_time_fit = time.perf_counter()
                time_used = end_time_fit - start_time_fit
                temp_fit_list.append(best_fitness)
                temp_time_list.append(time_used)
                curve_output2 = np.zeros(m)
                temp_len = len(curve_output)
                curve_output2[0:temp_len] = curve_output
                temp_curve_list.append(curve_output2)
        avg_fit = np.average(np.array(temp_fit_list))
        max_fit = np.max(np.array(temp_fit_list))
        min_fit = np.min(np.array(temp_fit_list))
        avg_time = np.average(np.array(temp_time_list))
        if curve == True:
            avg_curve_pre = np.vstack(temp_curve_list)
            avg_curve = np.mean(avg_curve_pre,axis=0)
            curve_output_list.append(avg_curve)
        avg_fit_list.append(avg_fit)
        max_fit_list.append(max_fit)
        min_fit_list.append(min_fit)
        avg_time_list.append(avg_time)
    results_df = pd.DataFrame({'max_iter':max_iters_list, 'avg_fit':avg_fit_list, 'max_fit':max_fit_list\
    ,'min_fit':min_fit_list, "avg_time":avg_time_list})
    if curve == False:
        return results_df
    else:
        return results_df, curve_output_list


def GA_best_params(problem, params_dict, inverse_fitness=False):
    pop_size_list = params_dict['pop_size']
    mutation_prob_list = params_dict['mutation_prob']
    max_attempts_list = params_dict['max_attempts']
    max_iters_list = params_dict['max_iters']
    max_iters_list = [max_iters_list[-1]]
    rand_list = params_dict['random_state']
    best_params_df = pd.DataFrame({'max_iter':[0], 'avg_fit':[999], 'max_fit':[999], 'min_fit':[999], 'avg_time':[0]\
    , 'pop_size':[0],'mutation_prob':[0], 'max_attempts':[0]})
    for p in pop_size_list:
        for mp in mutation_prob_list:
            for ma in max_attempts_list:
                temp_results_df = fitness_by_iter('GA', problem,  max_iters_list, rand_list, pop_size=p\
                ,mutation_prob=mp, max_attempts=ma, curve=False)
                temp_results_df['pop_size'] = p
                temp_results_df['mutation_prob'] = mp
                temp_results_df['max_attempts'] = ma
                best_params_df = pd.concat([best_params_df, temp_results_df]).copy()
    best_params_df = best_params_df[(best_params_df['max_iter'] > 0)].copy()
    if inverse_fitness == True:
        best_params_df['avg_fit'] = best_params_df['avg_fit'].apply(lambda x: 1/x)
        best_params_df['max_fit'] = best_params_df['max_fit'].apply(lambda x: 1/x)
    best_params_df.sort_values('avg_fit', ascending=False)
    return best_params_df


def SA_best_params(problem, params_dict, inverse_fitness=False):
    init_temp_list = params_dict['init_temp']
    decay_list = params_dict['decay']
    max_attempts_list = params_dict['max_attempts']
    max_iters_list = params_dict['max_iters']
    max_iters_list = [max_iters_list[-1]]
    rand_list = params_dict['random_state']
    best_params_df = pd.DataFrame({'max_iter':[0], 'avg_fit':[999], 'max_fit':[999], 'min_fit':[999], 'avg_time':[0]\
    , 'init_temp':[0],'decay':[0], 'max_attempts':[0]})
    for t in init_temp_list:
        for d in decay_list:
            for ma in max_attempts_list:
                temp_schedule = mlrose.GeomDecay(init_temp=t, decay=d)
                temp_results_df = fitness_by_iter('SA', problem, max_iters_list, rand_list, schedule=temp_schedule\
                , init_state=None, max_attempts=ma, curve=False)
                temp_results_df['init_temp'] = t
                temp_results_df['decay'] = d
                temp_results_df['max_attempts'] = ma
                best_params_df = pd.concat([best_params_df, temp_results_df]).copy()
    best_params_df = best_params_df[(best_params_df['max_iter'] > 0)].copy()
    if inverse_fitness == True:
        best_params_df['avg_fit'] = best_params_df['avg_fit'].apply(lambda x: 1/x)
        best_params_df['max_fit'] = best_params_df['max_fit'].apply(lambda x: 1/x)
    best_params_df.sort_values('avg_fit', ascending=False)
    return best_params_df


def RHC_best_params(problem, params_dict, inverse_fitness=False):
    max_attempts_list = params_dict['max_attempts']
    restarts_list = params_dict['restarts']
    max_iters_list = params_dict['max_iters']
    max_iters_list = [max_iters_list[-1]]
    rand_list = params_dict['random_state']
    best_params_df = pd.DataFrame({'max_iter':[0], 'avg_fit':[999], 'max_fit':[999], 'min_fit':[999], 'avg_time':[0]\
    , 'restarts':[0],'max_attempts':[0]})
    for r in restarts_list:
        for ma in max_attempts_list:
            temp_results_df = fitness_by_iter('RHC', problem, max_iters_list, rand_list, max_attempts=ma, restarts=r\
            ,init_state=None, curve=False)
            temp_results_df['restarts'] = r
            temp_results_df['max_attempts'] = ma
            best_params_df = pd.concat([best_params_df, temp_results_df]).copy()
    best_params_df = best_params_df[(best_params_df['max_iter'] > 0)].copy()
    if inverse_fitness == True:
        best_params_df['avg_fit'] = best_params_df['avg_fit'].apply(lambda x: 1/x)
        best_params_df['max_fit'] = best_params_df['max_fit'].apply(lambda x: 1/x)
    best_params_df.sort_values('avg_fit', ascending=False)
    return best_params_df


def MIMIC_best_params(problem, params_dict, inverse_fitness=False):
    pop_size_list = params_dict['pop_size']
    keep_pct_list = params_dict['keep_pct']
    max_attempts_list = params_dict['max_attempts']
    max_iters_list = params_dict['max_iters']
    #max_iters_list = [max_iters_list[-1]]
    rand_list = params_dict['random_state']
    best_params_df = pd.DataFrame({'max_iter':[0], 'avg_fit':[999], 'max_fit':[999], 'min_fit':[999], 'avg_time':[0]\
    , 'pop_size':[0],'keep_pct':[0], 'max_attempts':[0]})
    for p in pop_size_list:
        for k in keep_pct_list:
            for ma in max_attempts_list:
                for iter in max_iters_list:
                    temp_results_df = fitness_by_iter('MIMIC', problem,  max_iters_list, rand_list, pop_size=p\
                    ,keep_pct=k, max_attempts=ma, curve=False, fast_mimic=True)
                    temp_results_df['pop_size'] = p
                    temp_results_df['keep_pct'] = k
                    temp_results_df['max_attempts'] = ma
                    best_params_df = pd.concat([best_params_df, temp_results_df]).copy()
    best_params_df = best_params_df[(best_params_df['max_iter'] > 0)].copy()
    if inverse_fitness == True:
        best_params_df['avg_fit'] = best_params_df['avg_fit'].apply(lambda x: 1/x)
        best_params_df['max_fit'] = best_params_df['max_fit'].apply(lambda x: 1/x)
    best_params_df.sort_values('avg_fit', ascending=False)
    return best_params_df


def create_TSP(space_dim, cities, seed = 12345, return_lists_too=False):
    random.seed(seed)
    x_coordinates = []
    y_coordinates = []
    for i in range(cities):
        x = random.randint(0, space_dim)
        y = random.randint(0, space_dim)
        x_coordinates.append(x)
        y_coordinates.append(y)
    coords = list(zip(x_coordinates, y_coordinates))
    if return_lists_too == True:
        return coords, x_coordinates, y_coordinates
    return coords


def create_Knapsack(length, seed = 12345, max_val=50):
    random.seed(seed)
    weights = []
    values = []
    for i in range(length):
        w = random.randint(1, max_val)
        v = random.randint(1, max_val)
        weights.append(w)
        values.append(v)
    return weights, values


def curve_to_df(curve_output, max):
    temp_array = np.zeros(max)
    temp_len = len(curve_output)
    temp_array[0:temp_len] = curve_output
    if temp_len < max:
        last_value = curve_output[-1]
        temp_array[temp_len:max] = last_value
    temp_df = pd.DataFrame(temp_array)
    index_list = [1,3,7,15,31,63,127]
    short_temp_array = temp_array[index_list]
    temp_df2 = pd.DataFrame(short_temp_array)
    return temp_df, temp_df2


def main():
    ## SET SOME PARAMS TO USE GLOBALLY
    max_iters_list = [50,100,1000] #,32,64,128,256,512,1024]
    max_iters_list_full = [2,4,8,16,32,64,128,256,512,1024]
    rand_list = [1,11,22] #,44,55,66,77,88,99]
    rand_list_full = [0,11,22,33,44,55,66,77,88,99]
    input_location = 'data/'
    output_location = 'outputs/'
    chart_output_location = 'charts/'
    prefix = '5th_'

    ## DEFINE PROBLEMS TO SOLVE
    # Traveling Salesman Problem (TSP)
    space_length = 1000
    cities_cnt = 200
    coords_list, x, y = create_TSP(space_length, cities_cnt, return_lists_too=True)
    plt.plot(x, y, 'o')
    plt.savefig(chart_output_location+'TPS_visual'+'.png')
    fitness_coords = mlrose.TravellingSales(coords = coords_list)
    problem_TSP = mlrose.TSPOpt(length = len(coords_list), fitness_fn = fitness_coords, maximize=False)

    # 4 Peaks
    t_pct = 0.1
    length = 200
    fitness_4_peaks = mlrose.FourPeaks(t_pct=t_pct)
    problem_4P = mlrose.DiscreteOpt(length = length, fitness_fn = fitness_4_peaks, maximize = True, max_val = 2)
    problem_4P_small = mlrose.DiscreteOpt(length = 50, fitness_fn = fitness_4_peaks, maximize = True, max_val = 2)
    problem_4P_big = mlrose.DiscreteOpt(length = 1000, fitness_fn = fitness_4_peaks, maximize = True, max_val = 2)

    # Continuous Peaks
    t_pct = 0.1
    length = 200
    fitness_cont_peaks = mlrose.ContinuousPeaks(t_pct=t_pct)
    problem_cont_peaks = mlrose.DiscreteOpt(length = length, fitness_fn = fitness_cont_peaks, maximize = True, max_val = 2)

    # Flip Flop
    length = 200
    fitness_FF = mlrose.FlipFlop()
    problem_FF = mlrose.DiscreteOpt(length = length, fitness_fn = fitness_FF, maximize = True, max_val = 2)
    problem_FF_small = mlrose.DiscreteOpt(length = 50, fitness_fn = fitness_FF, maximize = True, max_val = 2)
    problem_FF_big = mlrose.DiscreteOpt(length = 1000, fitness_fn = fitness_FF, maximize = True, max_val = 2)


    # Knapsack
    length = 200
    weights, values = create_Knapsack(length)
    weights_big, values_big = create_Knapsack(1000)
    weights_small, values_small = create_Knapsack(50)
    fitness_KS = mlrose.Knapsack(weights, values, max_weight_pct=0.65)
    fitness_KS_big = mlrose.Knapsack(weights_big, values_big, max_weight_pct=0.65)
    fitness_KS_small = mlrose.Knapsack(weights_small, values_small, max_weight_pct=0.65)
    problem_KS = mlrose.DiscreteOpt(length = length, fitness_fn = fitness_KS, maximize = True, max_val = 2)
    problem_KS_big = mlrose.DiscreteOpt(length = 1000, fitness_fn = fitness_KS_big, maximize = True, max_val = 2)
    problem_KS_small = mlrose.DiscreteOpt(length = 50, fitness_fn = fitness_KS_small, maximize = True, max_val = 2)

    dict_of_param_dict = {}
    dict_of_param_dict['GA'] = {
    'pop_size':[100,200],#,1000],
    'mutation_prob':[0.5, 0.1, 0.2],
    'max_attempts':[5,10,30],
    'max_iters':max_iters_list,
    'random_state':rand_list
    }
    dict_of_param_dict['RHC'] = {
    'max_attempts':[30,50,100],  #[5,10,20,50]
    'restarts':[5,10,20],  #[0,1,2,5]
    'max_iters':max_iters_list,
    'random_state':rand_list
    }
    dict_of_param_dict['SA'] = {
    'max_attempts':[10,50,100],
    'init_temp':[1.0,10.0,0.5,20,100,1000],
    'decay':[0.99,0.8,0.5],
    'max_iters':max_iters_list,
    'random_state':rand_list
    }
    dict_of_param_dict['MIMIC'] = {
    'pop_size':[100,150],
    'keep_pct':[0.5,0.2],
    'max_attempts':[10],
    'max_iters':[100],
    'random_state':rand_list
    }

    MIMIC_FF = {
    'pop_size':100,
    'keep_pct':0.5,
    'max_attempts':30,
    'max_iters':[2,4,8,16,32,64,128],  ## put full list here before uploading
    'random_state':[0,11,22,33,44]
    }
    MIMIC_4P = {
    'pop_size':150,
    'keep_pct':0.2,
    'max_attempts':30,
    'max_iters':[2,4,8,16,32,64,128],  ## put full list here before uploading
    'random_state':[0,11,22,33,44]
    }
    MIMIC_KS = {
    'pop_size':150,
    'keep_pct':0.5,
    'max_attempts':30,
    'max_iters':[2,4,8,16,32,64,128],  ## put full list here before uploading
    'random_state':[0,11,22,33,44]
    }
    MIMIC_CP = {
    'pop_size':200,
    'keep_pct':0.2,
    'max_attempts':30,
    'max_iters':[2,4,8,16,32,64,128],  ## put full list here before uploading
    'random_state':[0,11,22,33,44]
    }
    GA_FF = {
    'pop_size':200,#,1000],
    'mutation_prob':0.5,
    'max_attempts':30,
    'max_iters':max_iters_list_full,
    'random_state':rand_list_full
    }

    MIMIC_FF2 = {
    'pop_size':[100],
    'keep_pct':[0.5],
    'max_attempts':[30,50],
    'max_iters':[64],
    'random_state':[55] #,66,77,88,99]
    }

    print("starting MIMIC FF")
    # GETTING MIMIC FF RESULTS
    print("starting MIMIC FF...")
    ''' ## Started running at 3am
    results_df, curve_output_list = fitness_by_iter('MIMIC', problem_FF, MIMIC_FF['max_iters'], MIMIC_FF['random_state']\
    , pop_size=MIMIC_FF['pop_size'], max_attempts=MIMIC_FF['max_attempts'], curve=True, keep_pct=MIMIC_FF['keep_pct'])
    results_df.to_csv(output_location + 'final_MIMIC_FF_attempt_3am.csv')


    results_df, curve_output_list = fitness_by_iter('MIMIC', problem_4P, MIMIC_4P['max_iters'], MIMIC_4P['random_state']\
    , pop_size=MIMIC_4P['pop_size'], max_attempts=MIMIC_4P['max_attempts'], curve=True, keep_pct=MIMIC_4P['keep_pct'])
    results_df.to_csv(output_location + 'final_MIMIC_4P_attempt_3am.csv')


    results_df, curve_output_list = fitness_by_iter('MIMIC', problem_KS, MIMIC_KS['max_iters'], MIMIC_KS['random_state']\
    , pop_size=MIMIC_KS['pop_size'], max_attempts=MIMIC_KS['max_attempts'], curve=True, keep_pct=MIMIC_KS['keep_pct'])
    results_df.to_csv(output_location + 'final_MIMIC_KS_attempt_3am.csv')


    results_df, curve_output_list = fitness_by_iter('MIMIC', problem_cont_peaks, MIMIC_CP['max_iters'], MIMIC_CP['random_state']\
    , pop_size=MIMIC_CP['pop_size'], max_attempts=MIMIC_CP['max_attempts'], curve=True, keep_pct=MIMIC_CP['keep_pct'])
    results_df.to_csv(output_location + 'final_MIMIC_CP_attempt_3am.csv')

    '''














    ## USED FOR GRID SEARCHING PARAMETERS FOR RO ON 3 PROBLEMS
    GA_params_dict = get_params_for_grid_search('GA', max_iters_list=[200])
    print("Here are my GA params for grid search: ", GA_params_dict)
    SA_params_dict = get_params_for_grid_search('SA', max_iters_list=max_iters_list)
    print("Here are my SA params for grid search: ", SA_params_dict)
    RHC_params_dict = get_params_for_grid_search('RHC', max_iters_list=max_iters_list)
    print("Here are my RHC params for grid search: ", RHC_params_dict)
    MIMIC_params_dict = get_params_for_grid_search('MIMIC', max_iters_list=max_iters_list)
    print("Here are my MIMIC params for grid search: ", MIMIC_params_dict)
    #grid_search_MIMIC = MIMIC_best_params(problem_TPS, MIMIC_params_dict, inverse_fitness=False)
    #grid_search_MIMIC.to_csv(output_location + 'grid_search_MIMIC.csv')
    '''
    grid_search_GA = GA_best_params(problem_FF, GA_params_dict, inverse_fitness=False)
    grid_search_GA.to_csv(output_location + prefix + 'grid_search_GA_FF_really.csv')
    print("finished GA")
    grid_search_MIMIC = MIMIC_best_params(problem_FF, MIMIC_params_dict, inverse_fitness=False)
    grid_search_MIMIC.to_csv(output_location + prefix + 'grid_search_MIMIC_FF_really.csv')
    '''
    print("finished MIMIC FF")


    print("Doing GA rn")
    #results_df, curve_output_list = fitness_by_iter('GA', problem_FF, GA_FF['max_iters'], GA_FF['random_state']\
    #, pop_size=GA_FF['pop_size'], max_attempts=GA_FF['max_attempts'], mutation_prob=GA_FF['mutation_prob'],curve=True)
    #results_df.to_csv(output_location + 'final_MIMIC_FF_attempt_1am.csv')
    print("finished GA")

    ''' GRID SEARCHING

    print("Starting grid search for RHC")
    grid_search_RHC = RHC_best_params(problem_TSP, RHC_params_dict, inverse_fitness=False)
    grid_search_RHC.to_csv(output_location + prefix +'grid_search_RHC_TSP.csv')
    grid_search_RHC = RHC_best_params(problem_FF, RHC_params_dict, inverse_fitness=False)
    grid_search_RHC.to_csv(output_location + prefix + 'grid_search_RHC_FF.csv')
    grid_search_RHC = RHC_best_params(problem_cont_peaks, RHC_params_dict, inverse_fitness=False)
    grid_search_RHC.to_csv(output_location + prefix + 'grid_search_RHC_cont_peaks.csv')
    grid_search_RHC = RHC_best_params(problem_4P, RHC_params_dict, inverse_fitness=False)
    grid_search_RHC.to_csv(output_location + prefix + 'grid_search_RHC_4P.csv')

    print("Starting grid search for SA")
    grid_search_SA = SA_best_params(problem_TSP, SA_params_dict, inverse_fitness=False)
    grid_search_SA.to_csv(output_location + prefix + 'grid_search_SA_TSP.csv')
    grid_search_SA = SA_best_params(problem_FF, SA_params_dict, inverse_fitness=False)
    grid_search_SA.to_csv(output_location + prefix + 'grid_search_SA_FF.csv')
    grid_search_SA = SA_best_params(problem_cont_peaks, SA_params_dict, inverse_fitness=False)
    grid_search_SA.to_csv(output_location + prefix + 'grid_search_SA_cont_peaks.csv')
    grid_search_SA = SA_best_params(problem_4P, SA_params_dict, inverse_fitness=False)
    grid_search_SA.to_csv(output_location + prefix + 'grid_search_SA_4P.csv')

    print("Starting grid search for GA")
    grid_search_GA = GA_best_params(problem_TSP, GA_params_dict, inverse_fitness=False)
    grid_search_GA.to_csv(output_location + prefix + 'grid_search_GA_TSP.csv')
    grid_search_GA = GA_best_params(problem_FF, GA_params_dict, inverse_fitness=False)
    grid_search_GA.to_csv(output_location + prefix + 'grid_search_GA_FF.csv')
    grid_search_GA = GA_best_params(problem_cont_peaks, GA_params_dict, inverse_fitness=False)
    grid_search_GA.to_csv(output_location + prefix + 'grid_search_GA_cont_peaks.csv')
    grid_search_GA = GA_best_params(problem_4P, GA_params_dict, inverse_fitness=False)
    grid_search_GA.to_csv(output_location + prefix + 'grid_search_GA_4P.csv')
    '''

    '''
    print("Starting grid search for MIMIC")
    grid_search_MIMIC = MIMIC_best_params(problem_FF, MIMIC_params_dict, inverse_fitness=False)
    grid_search_MIMIC.to_csv(output_location + prefix + 'grid_search_MIMIC_FF.csv')
    #grid_search_MIMIC = MIMIC_best_params(problem_cont_peaks, MIMIC_params_dict, inverse_fitness=False)
    #grid_search_MIMIC.to_csv(output_location + prefix + 'grid_search_MIMIC_cont_peaks.csv')
    grid_search_MIMIC = MIMIC_best_params(problem_4P, MIMIC_params_dict, inverse_fitness=False)
    grid_search_MIMIC.to_csv(output_location + prefix + 'grid_search_MIMIC_4P.csv')
    #grid_search_MIMIC = MIMIC_best_params(problem_TSP, MIMIC_params_dict, inverse_fitness=False)
    #grid_search_MIMIC.to_csv(output_location + 'grid_search_MIMIC_TSP.csv')
    print("Finished MIMIC grid searches")

    print("Starting grid search for Knapsack")
    #grid_search_MIMIC = MIMIC_best_params(problem_KS, MIMIC_params_dict, inverse_fitness=False)
    #grid_search_MIMIC.to_csv(output_location + prefix + 'grid_search_MIMIC_KS.csv')
    #grid_search_GA = GA_best_params(problem_KS, GA_params_dict, inverse_fitness=False)
    #grid_search_GA.to_csv(output_location + prefix + 'grid_search_GA_KS.csv')
    grid_search_SA = SA_best_params(problem_KS, SA_params_dict, inverse_fitness=False)
    grid_search_SA.to_csv(output_location + prefix + 'grid_search_SA_KS.csv')
    grid_search_RHC = RHC_best_params(problem_KS, RHC_params_dict, inverse_fitness=False)
    grid_search_RHC.to_csv(output_location + prefix + 'grid_search_RHC_KS.csv')
    '''



    ## Fitting MIMIC separately and with fewer iterations for all except the FF as run time is so long for MIMIC
    max=128
    ''' MIMIC CURVE FOR CHARTS ##### Started (again) at 8am ######

    print("Fitting for MIMIC using the 'curve=True' functionality")
    print("First for KS")
    start_time_fit = time.perf_counter()
    a,b,curve_output = mlrose.mimic(problem_KS, pop_size=100, keep_pct=0.5, max_attempts=10, max_iters=128, curve=True\
    , random_state=0)
    end_time_fit = time.perf_counter()
    time_used = end_time_fit - start_time_fit
    df1, df2 = curve_to_df(curve_output, max)
    df2['time_to_128'] = time_used
    df1.to_csv(output_location+'MIMIC_KS_full_curve.csv')
    df2.to_csv(output_location+'MIMIC_KS_short_curve.csv')
    print("Finished KS")

    print("Next for 4 Peaks")
    start_time_fit = time.perf_counter()
    a,b,curve_output = mlrose.mimic(problem_4P, pop_size=150, keep_pct=0.2, max_attempts=10, max_iters=128, curve=True\
    , random_state=0)
    end_time_fit = time.perf_counter()
    time_used = end_time_fit - start_time_fit
    df1, df2 = curve_to_df(curve_output, max)
    df2['time_to_128'] = time_used
    df1.to_csv(output_location+'MIMIC_4P_full_curve.csv')
    df2.to_csv(output_location+'MIMIC_4P_short_curve.csv')
    print("Finished 4 Peaks")

    print("Next for 4 Peaks with 100 and 0.5")
    start_time_fit = time.perf_counter()
    a,b,curve_output = mlrose.mimic(problem_4P, pop_size=100, keep_pct=0.5, max_attempts=10, max_iters=128, curve=True\
    , random_state=0)
    end_time_fit = time.perf_counter()
    time_used = end_time_fit - start_time_fit
    df1, df2 = curve_to_df(curve_output, max)
    df2['time_to_128'] = time_used
    df1.to_csv(output_location+'MIMIC_4P_pop100_keep50_full_curve.csv')
    df2.to_csv(output_location+'MIMIC_4P_pop100_keep50_short_curve.csv')
    print("Finished 4 Peaks")

    print("Next for 4 Peaks with 100 and 0.2")
    start_time_fit = time.perf_counter()
    a,b,curve_output = mlrose.mimic(problem_4P, pop_size=100, keep_pct=0.2, max_attempts=10, max_iters=128, curve=True\
    , random_state=0)
    end_time_fit = time.perf_counter()
    time_used = end_time_fit - start_time_fit
    df1, df2 = curve_to_df(curve_output, max)
    df2['time_to_128'] = time_used
    df1.to_csv(output_location+'MIMIC_4P_pop100_keep20_full_curve.csv')
    df2.to_csv(output_location+'MIMIC_4P_pop100_keep20_short_curve.csv')
    print("Finished 4 Peaks")

    print("Next for 4 Peaks with 150 and 0.5")
    start_time_fit = time.perf_counter()
    a,b,curve_output = mlrose.mimic(problem_4P, pop_size=100, keep_pct=0.2, max_attempts=10, max_iters=128, curve=True\
    , random_state=0)
    end_time_fit = time.perf_counter()
    time_used = end_time_fit - start_time_fit
    df1, df2 = curve_to_df(curve_output, max)
    df2['time_to_128'] = time_used
    df1.to_csv(output_location+'MIMIC_4P_pop150_keep50_full_curve.csv')
    df2.to_csv(output_location+'MIMIC_4P_pop150_keep50_short_curve.csv')
    print("Finished 4 Peaks")

    print("Next for 4 Peaks Big")
    start_time_fit = time.perf_counter()
    a,b,curve_output = mlrose.mimic(problem_4P_big, pop_size=150, keep_pct=0.2, max_attempts=10, max_iters=128, curve=True\
    , random_state=0)
    end_time_fit = time.perf_counter()
    time_used = end_time_fit - start_time_fit
    df1, df2 = curve_to_df(curve_output, max)
    df2['time_to_128'] = time_used
    df1.to_csv(output_location+'MIMIC_4P_big_full_curve.csv')
    df2.to_csv(output_location+'MIMIC_4P_big_short_curve.csv')
    print("Finished 4 Peaks Big")

    print("Next for KS Small")
    start_time_fit = time.perf_counter()
    a,b,curve_output = mlrose.mimic(problem_KS_small, pop_size=100, keep_pct=0.5, max_attempts=10, max_iters=128, curve=True\
    , random_state=0)
    end_time_fit = time.perf_counter()
    time_used = end_time_fit - start_time_fit
    df1, df2 = curve_to_df(curve_output, max)
    df2['time_to_128'] = time_used
    df1.to_csv(output_location+'MIMIC_KS_small_full_curve.csv')
    df2.to_csv(output_location+'MIMIC_KS_small_short_curve.csv')
    print("Finished KS small")

    print("Next FF small")
    start_time_fit = time.perf_counter()
    a,b,curve_output = mlrose.mimic(problem_FF_small, pop_size=100, keep_pct=0.5, max_attempts=10, max_iters=128, curve=True\
    , random_state=0)
    end_time_fit = time.perf_counter()
    time_used = end_time_fit - start_time_fit
    df1, df2 = curve_to_df(curve_output, max)
    df2['time_to_128'] = time_used
    df1.to_csv(output_location+'MIMIC_FF_small_full_curve.csv')
    df2.to_csv(output_location+'MIMIC_KS_small_short_curve.csv')
    print("Finished FF Small")

    print("Next for 4 Peaks Small")
    start_time_fit = time.perf_counter()
    a,b,curve_output = mlrose.mimic(problem_4P_small, pop_size=150, keep_pct=0.2, max_attempts=10, max_iters=128, curve=True\
    , random_state=0)
    end_time_fit = time.perf_counter()
    time_used = end_time_fit - start_time_fit
    df1, df2 = curve_to_df(curve_output, max)
    df2['time_to_128'] = time_used
    df1.to_csv(output_location+'MIMIC_4P_small_full_curve.csv')
    df2.to_csv(output_location+'MIMIC_4P_small_short_curve.csv')
    print("Finished 4 Peaks Small")
    '''

    ### Now GA


    GA_FF = {
    'pop_size':100,#,1000],
    'mutation_prob':0.1,
    'max_attempts':30,
    'max_iters':max_iters_list_full,
    'random_state':rand_list_full
    }
    GA_KS = {
    'pop_size':200,#,1000],
    'mutation_prob':0.2,
    'max_attempts':30,
    'max_iters':max_iters_list_full,
    'random_state':rand_list_full
    }
    GA_4P = {
    'pop_size':200,#,1000],
    'mutation_prob':0.5,
    'max_attempts':30,
    'max_iters':max_iters_list_full,
    'random_state':rand_list_full
    }
    ''' More fitness by iteration calculations
    #results_df, curve_output_list = fitness_by_iter('GA', problem_FF, GA_FF['max_iters'], GA_FF['random_state']\
    #, pop_size=GA_FF['pop_size'], max_attempts=GA_FF['max_attempts'], curve=True, mutation_prob=GA_FF['mutation_prob'])
    #results_df.to_csv(output_location + 'final_GA_FF_attempt_8am.csv')
    results_df, curve_output_list = fitness_by_iter('GA', problem_FF_small, GA_FF['max_iters'], GA_FF['random_state']\
    , pop_size=GA_FF['pop_size'], max_attempts=GA_FF['max_attempts'], curve=True, mutation_prob=GA_FF['mutation_prob'])
    results_df.to_csv(output_location + 'final_GA_FF_small_attempt_8am.csv')
    results_df, curve_output_list = fitness_by_iter('GA', problem_FF_big, GA_FF['max_iters'], GA_FF['random_state']\
    , pop_size=GA_FF['pop_size'], max_attempts=GA_FF['max_attempts'], curve=True, mutation_prob=GA_FF['mutation_prob'])
    results_df.to_csv(output_location + 'final_GA_FF_big_attempt_8am.csv')



    #results_df, curve_output_list = fitness_by_iter('GA', problem_4P, GA_4P['max_iters'], GA_4P['random_state']\
    #, pop_size=GA_4P['pop_size'], max_attempts=GA_4P['max_attempts'], curve=True, mutation_prob=GA_4P['mutation_prob'])
    #results_df.to_csv(output_location + 'final_GA_4P_attempt_8am.csv')
    results_df, curve_output_list = fitness_by_iter('GA', problem_4P_big, GA_4P['max_iters'], GA_4P['random_state']\
    , pop_size=GA_4P['pop_size'], max_attempts=GA_4P['max_attempts'], curve=True, mutation_prob=GA_4P['mutation_prob'])
    results_df.to_csv(output_location + 'final_GA_4P_big_attempt_8am.csv')
    results_df, curve_output_list = fitness_by_iter('GA', problem_4P_small, GA_4P['max_iters'], GA_4P['random_state']\
    , pop_size=GA_4P['pop_size'], max_attempts=GA_4P['max_attempts'], curve=True, mutation_prob=GA_4P['mutation_prob'])
    results_df.to_csv(output_location + 'final_GA_4P_small_attempt_8am.csv')



    #results_df, curve_output_list = fitness_by_iter('GA', problem_KS, GA_KS['max_iters'], GA_KS['random_state']\
    #, pop_size=GA_KS['pop_size'], max_attempts=GA_KS['max_attempts'], curve=True, mutation_prob=GA_KS['mutation_prob'])
    #results_df.to_csv(output_location + 'final_GA_KS_attempt_8am.csv')
    results_df, curve_output_list = fitness_by_iter('GA', problem_KS_big, GA_KS['max_iters'], GA_KS['random_state']\
    , pop_size=GA_KS['pop_size'], max_attempts=GA_KS['max_attempts'], curve=True, mutation_prob=GA_KS['mutation_prob'])
    results_df.to_csv(output_location + 'final_GA_KS_big_attempt_8am.csv')
    results_df, curve_output_list = fitness_by_iter('GA', problem_KS_small, GA_KS['max_iters'], GA_KS['random_state']\
    , pop_size=GA_KS['pop_size'], max_attempts=GA_KS['max_attempts'], curve=True, mutation_prob=GA_KS['mutation_prob'])
    results_df.to_csv(output_location + 'final_GA_KS_small_attempt_8am.csv')

    '''


    ########### SA
    print("now doing SA")
    SA_4P = {
    'max_attempts':10,
    'schedule':mlrose.GeomDecay(init_temp=100, decay=0.8),
    'max_iters':max_iters_list_full,
    'random_state':rand_list_full
    }

    SA_FF = {
    'max_attempts':10,
    'schedule':mlrose.GeomDecay(init_temp=100, decay=0.8),
    'max_iters':max_iters_list_full,
    'random_state':rand_list_full
    }

    results_df, curve_output_list = fitness_by_iter('SA', problem_4P, SA_4P['max_iters'], SA_4P['random_state']\
    , schedule=SA_4P['schedule'], max_attempts=SA_4P['max_attempts'], curve=True)
    results_df.to_csv(output_location + 'final_SA_4P_attempt_8am.csv')

    results_df, curve_output_list = fitness_by_iter('SA', problem_4P_big, SA_4P['max_iters'], SA_4P['random_state']\
    , schedule=SA_4P['schedule'], max_attempts=SA_4P['max_attempts'], curve=True)
    results_df.to_csv(output_location + 'final_SA_4P_big_attempt_8am.csv')
    results_df, curve_output_list = fitness_by_iter('SA', problem_4P, SA_4P['max_iters'], SA_4P['random_state']\
    , schedule=SA_4P['schedule'], max_attempts=SA_4P['max_attempts'], curve=True)
    results_df.to_csv(output_location + 'final_SA_4P_small_attempt_8am.csv')

    ''' more fitness by iteration calculations
    #results_df, curve_output_list = fitness_by_iter('SA', problem_FF, SA_FF['max_iters'], SA_FF['random_state']\
    #, schedule=SA_FF['schedule'], max_attempts=SA_FF['max_attempts'], curve=True)
    #results_df.to_csv(output_location + 'final_SA_FF_attempt_8am.csv')
    results_df, curve_output_list = fitness_by_iter('SA', problem_FF_big, SA_FF['max_iters'], SA_FF['random_state']\
    , schedule=SA_FF['schedule'], max_attempts=SA_FF['max_attempts'], curve=True)
    results_df.to_csv(output_location + 'final_SA_FF_big_attempt_8am.csv')
    results_df, curve_output_list = fitness_by_iter('SA', problem_FF_small, SA_FF['max_iters'], SA_FF['random_state']\
    , schedule=SA_FF['schedule'], max_attempts=SA_FF['max_attempts'], curve=True)
    results_df.to_csv(output_location + 'final_SA_FF_small_attempt_8am.csv')


    SA_4P = {
    'max_attempts':10,
    'schedule':mlrose.GeomDecay(init_temp=100, decay=0.8),
    'max_iters':max_iters_list_full,
    'random_state':rand_list_full
    }

    results_df, curve_output_list = fitness_by_iter('KS', problem_4P, SA_4P['max_iters'], SA_4P['random_state']\
    , schedule=SA_4P['schedule'], max_attempts=SA_4P['max_attempts'], curve=True)
    results_df.to_csv(output_location + 'final_SA_4P_attempt_8am.csv')
    results_df, curve_output_list = fitness_by_iter('SA', problem_4P_big, SA_4P['max_iters'], SA_4P['random_state']\
    , schedule=SA_4P['schedule'], max_attempts=SA_4P['max_attempts'], curve=True)
    results_df.to_csv(output_location + 'final_SA_4P_big_attempt_8am.csv')
    results_df, curve_output_list = fitness_by_iter('SA', problem_4P, SA_4P['max_iters'], SA_4P['random_state']\
    , schedule=SA_4P['schedule'], max_attempts=SA_4P['max_attempts'], curve=True)
    results_df.to_csv(output_location + 'final_SA_4P_small_attempt_8am.csv')
    '''
    print("picking up where I left off on making the final curves..")

    SA_KS = {
    'max_attempts':10,
    'schedule':mlrose.GeomDecay(init_temp=1000, decay=0.99),
    'max_iters':max_iters_list_full,
    'random_state':rand_list_full
    }
    ''' more fitness by iteration calculations
    results_df, curve_output_list = fitness_by_iter('SA', problem_KS, SA_KS['max_iters'], SA_KS['random_state']\
    , schedule=SA_KS['schedule'], max_attempts=SA_KS['max_attempts'], curve=True)
    results_df.to_csv(output_location + 'final_SA_KS_attempt_8am.csv')
    results_df, curve_output_list = fitness_by_iter('SA', problem_KS_big, SA_KS['max_iters'], SA_KS['random_state']\
    , schedule=SA_KS['schedule'], max_attempts=SA_KS['max_attempts'], curve=True)
    results_df.to_csv(output_location + 'final_SA_KS_big_attempt_8am.csv')
    results_df, curve_output_list = fitness_by_iter('SA', problem_KS_small, SA_KS['max_iters'], SA_KS['random_state']\
    , schedule=SA_KS['schedule'], max_attempts=SA_KS['max_attempts'], curve=True)
    results_df.to_csv(output_location + 'final_SA_KS_small_attempt_8am.csv')
    '''

    RHC_KS = {
    'max_attempts':50,
    'restarts':20,
    'max_iters':max_iters_list_full,
    'random_state':rand_list_full
    }
    '''
    results_df, curve_output_list = fitness_by_iter('RHC', problem_KS, RHC_KS['max_iters'], RHC_KS['random_state']\
    , restarts=RHC_KS['restarts'], max_attempts=RHC_KS['max_attempts'], curve=True)
    results_df.to_csv(output_location + 'final_RHS_KS_attempt_8am.csv')
    results_df, curve_output_list = fitness_by_iter('RHC', problem_KS_big, RHC_KS['max_iters'], RHC_KS['random_state']\
    , restarts=RHC_KS['restarts'], max_attempts=RHC_KS['max_attempts'], curve=True)
    results_df.to_csv(output_location + 'final_RHS_KS_big_attempt_8am.csv')
    results_df, curve_output_list = fitness_by_iter('RHC', problem_KS_small, RHC_KS['max_iters'], RHC_KS['random_state']\
    , restarts=RHC_KS['restarts'], max_attempts=RHC_KS['max_attempts'], curve=True)
    results_df.to_csv(output_location + 'final_RHS_KS_small_attempt_8am.csv')
    '''
    RHC_FF = {
    'max_attempts':50,
    'restarts':20,
    'max_iters':max_iters_list_full,
    'random_state':rand_list_full
    }
    '''
    results_df, curve_output_list = fitness_by_iter('RHC', problem_FF, RHC_FF['max_iters'], RHC_FF['random_state']\
    , restarts=RHC_FF['restarts'], max_attempts=RHC_FF['max_attempts'], curve=True)
    results_df.to_csv(output_location + 'final_RHS_FF_attempt_8am.csv')
    results_df, curve_output_list = fitness_by_iter('RHC', problem_FF_big, RHC_FF['max_iters'], RHC_FF['random_state']\
    , restarts=RHC_FF['restarts'], max_attempts=RHC_FF['max_attempts'], curve=True)
    results_df.to_csv(output_location + 'final_RHS_FF_big_attempt_8am.csv')
    results_df, curve_output_list = fitness_by_iter('RHC', problem_FF_small, RHC_FF['max_iters'], RHC_FF['random_state']\
    , restarts=RHC_FF['restarts'], max_attempts=RHC_FF['max_attempts'], curve=True)
    results_df.to_csv(output_location + 'final_RHS_FF_small_attempt_8am.csv')
    '''

    RHC_4P = {
    'max_attempts':50,
    'restarts':20,
    'max_iters':max_iters_list_full,
    'random_state':rand_list_full
    }

    '''
    results_df, curve_output_list = fitness_by_iter('RHC', problem_4P, RHC_4P['max_iters'], RHC_4P['random_state']\
    , restarts=RHC_4P['restarts'], max_attempts=RHC_4P['max_attempts'], curve=True)
    results_df.to_csv(output_location + 'final_RHS_4P_attempt_8am.csv')
    results_df, curve_output_list = fitness_by_iter('RHC', problem_4P_small, RHC_4P['max_iters'], RHC_4P['random_state']\
    , restarts=RHC_4P['restarts'], max_attempts=RHC_4P['max_attempts'], curve=True)
    results_df.to_csv(output_location + 'final_RHS_4P_small_attempt_8am.csv')
    results_df, curve_output_list = fitness_by_iter('RHC', problem_4P_big, RHC_4P['max_iters'], RHC_4P['random_state']\
    , restarts=RHC_4P['restarts'], max_attempts=RHC_4P['max_attempts'], curve=True)
    results_df.to_csv(output_location + 'final_RHS_4P_big_attempt_8am.csv')
    '''


    ## where it stopped
    print("I will now make the complexity curves for other algos")
    SA_4P_hacked = {
    'max_attempts':10,
    'schedule':mlrose.GeomDecay(init_temp=100, decay=0.99),
    'max_iters':max_iters_list_full,
    'random_state':rand_list_full
    }
    '''
    results_df, curve_output_list = fitness_by_iter('SA', problem_4P, SA_4P['max_iters'], SA_4P['random_state']\
    , schedule=SA_4P_hacked['schedule'], max_attempts=SA_4P['max_attempts'], curve=True)
    results_df.to_csv(output_location + 'final_SA_4P_decay_99.csv')

    results_df, curve_output_list = fitness_by_iter('SA', problem_4P, SA_4P['max_iters'], SA_4P['random_state']\
    , schedule=mlrose.GeomDecay(init_temp=1, decay=0.8), max_attempts=SA_4P['max_attempts'], curve=True)
    results_df.to_csv(output_location + 'final_SA_4P_T_1_decay_80.csv')

    results_df, curve_output_list = fitness_by_iter('GA', problem_KS, GA_KS['max_iters'], GA_KS['random_state']\
    , pop_size=GA_KS['pop_size'], max_attempts=GA_KS['max_attempts'], curve=True, mutation_prob=0.1)
    results_df.to_csv(output_location + 'final_GA_KS_mutation_01.csv')
    '''
    results_df, curve_output_list = fitness_by_iter('GA', problem_KS, GA_KS['max_iters'], GA_KS['random_state']\
    , pop_size=100, max_attempts=GA_KS['max_attempts'], curve=True, mutation_prob=0.2)
    results_df.to_csv(output_location + 'final_GA_KS_mutation_02_pop_100.csv')



    ## Need a few more MIMIC chart inputs
    #print("Need a few more MIMIC chart inputs, so I will now make those")
    #print("Next FF p=100 keep=0.2")
    ''' MIMIC inputs for charts
    start_time_fit = time.perf_counter()
    a,b,curve_output = mlrose.mimic(problem_FF, pop_size=100, keep_pct=0.2, max_attempts=10, max_iters=128, curve=True\
    , random_state=0)
    end_time_fit = time.perf_counter()
    time_used = end_time_fit - start_time_fit
    df1, df2 = curve_to_df(curve_output, max)
    df2['time_to_128'] = time_used
    df1.to_csv(output_location+'MIMIC_FF_p_100_k_20_full_curve.csv')
    df2.to_csv(output_location+'MIMIC_FF_p_100_k_20_short_curve.csv')
    print("Finished FF Big")

    print("Next FF p=150 keep=0.2")
    start_time_fit = time.perf_counter()
    a,b,curve_output = mlrose.mimic(problem_FF, pop_size=150, keep_pct=0.2, max_attempts=10, max_iters=128, curve=True\
    , random_state=0)
    end_time_fit = time.perf_counter()
    time_used = end_time_fit - start_time_fit
    df1, df2 = curve_to_df(curve_output, max)
    df2['time_to_128'] = time_used
    df1.to_csv(output_location+'MIMIC_FF_p_150_k_20_full_curve.csv')
    df2.to_csv(output_location+'MIMIC_FF_p_150_k_20_short_curve.csv')
    print("Finished FF Big")

    print("Next FF p=150 keep=0.5")
    start_time_fit = time.perf_counter()
    a,b,curve_output = mlrose.mimic(problem_FF, pop_size=150, keep_pct=0.5, max_attempts=10, max_iters=128, curve=True\
    , random_state=0)
    end_time_fit = time.perf_counter()
    time_used = end_time_fit - start_time_fit
    df1, df2 = curve_to_df(curve_output, max)
    df2['time_to_128'] = time_used
    df1.to_csv(output_location+'MIMIC_FF_p_150_k_50_full_curve.csv')
    df2.to_csv(output_location+'MIMIC_FF_p_150_k_50_short_curve.csv')
    print("Finished FF Big")

    print("First for KS Big")
    start_time_fit = time.perf_counter()
    a,b,curve_output = mlrose.mimic(problem_KS_big, pop_size=100, keep_pct=0.5, max_attempts=10, max_iters=128, curve=True\
    , random_state=0)
    end_time_fit = time.perf_counter()
    time_used = end_time_fit - start_time_fit
    df1, df2 = curve_to_df(curve_output, max)
    df2['time_to_128'] = time_used
    df1.to_csv(output_location+'MIMIC_KS_big_full_curve.csv')
    df2.to_csv(output_location+'MIMIC_KS_big_short_curve.csv')
    print("Finished KS Big")

    print("Next FF Big")
    start_time_fit = time.perf_counter()
    a,b,curve_output = mlrose.mimic(problem_FF_big, pop_size=100, keep_pct=0.5, max_attempts=10, max_iters=128, curve=True\
    , random_state=0)
    end_time_fit = time.perf_counter()
    time_used = end_time_fit - start_time_fit
    df1, df2 = curve_to_df(curve_output, max)
    df2['time_to_128'] = time_used
    df1.to_csv(output_location+'MIMIC_FF_big_full_curve.csv')
    df2.to_csv(output_location+'MIMIC_KS_big_short_curve.csv')
    print("Finished FF Big")
    '''













if __name__ == "__main__":
    main()
