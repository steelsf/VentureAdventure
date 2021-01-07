import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fill_nan_in_arrays(list_of_arrays):
    list_len = len(list_of_arrays)
    list_of_arrays_len = []
    for i in list_of_arrays:
        list_of_arrays_len.append(len(i))
    longest_array_len = max(list_of_arrays_len)
    new_array = np.empty((list_len,longest_array_len))
    new_array[:] = np.NaN
    index = 0
    for i in list_of_arrays:
        print(list_of_arrays_len)
        print("list_of_arrays_len[i]: ", list_of_arrays_len[index])
        new_array[index,0:list_of_arrays_len[index]] = i
        index = index + 1
    return new_array


def chart_curve(x_array, y_array, title, output_location, labels_list=0\
, use_f1=False, xlabel=None, ylabel=None, linewidth=1):
    plt.figure(figsize=(6.4*.5,4.8*.5))
    num_lines = y_array.shape[0]
    for i in range(num_lines):
        plt.plot(x_array, y_array[i], label = labels_list[i], linewidth=linewidth)
    plt.title(title, fontsize=10)
    plt.linewidth=4
    plt.grid(False)
    plt.xlabel(xlabel,fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.legend(fontsize=7)
    plt.savefig(output_location+title+'.png')
    plt.clf()
    return


def chart_bars(x_array, y_array, title, output_location, labels_list=0\
, use_f1=False, xlabel=None, ylabel=None):
    plt.close('all')
    plt.figure(figsize=(6.4*.5,4.8*.5))
    plt.bar(x_array, y_array)
    plt.title(title, fontsize=10)
    plt.grid(False)
    plt.xlabel(xlabel,fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    #plt.figure(figsize=(6.4*.5,4.8*.5))
    plt.tight_layout()
    plt.savefig(output_location+title+'.png')
    plt.clf()
    plt.close('all')
    return


def chart_hbars(x_array, y_array, title, output_location, labels_list=0\
, use_f1=False, xlabel=None, ylabel=None):
    plt.close('all')
    plt.figure(figsize=(6.4*.5,6.4*.5))
    plt.barh(x_array, y_array)
    plt.title(title, fontsize=10)
    plt.grid(False)
    plt.xlabel(xlabel,fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    #plt.figure(figsize=(6.4*.5,4.8*.5))
    plt.tight_layout()
    plt.savefig(output_location+title+'.png')
    plt.clf()
    plt.close('all')
    return


input_location = '/Users/vwy957/Documents/ML/markov/outputs/'
ql_chart_location = '/Users/vwy957/Documents/ML/markov/charts/QL_charts/'
_chart_location = '/Users/vwy957/Documents/ML/markov/charts/'


''' CHART TONS OF CONVERGENCE CHARTS FOR QL AND MAKE A REWARDS DF
'''
algorithm_list = ['QL ']
problem_list = [ 'Stir Crazy ','C19 Grid ',]
size_list = ['Small ', 'Medium ']#, 'Large ']
gamma_list = ['G=0.7 ', 'G=0.8 ', 'G=0.90 ', 'G=0.99 ']
epsilon_list = ['E=0.7 ','E=0.8 ', 'E=0.90 ','E=0.99 ']

folder = 'run_stats/'
f = '_run_stats_df.csv'
key_metrics_folder = 'single_key_metric/'
key_metrics_f = '_key_metrics_df.csv'
col_names=['Time', 'Iteration', 'reward','row_name','algorithm','problem','size','gamma','epsilon']
#concat_df = pd.DataFrame(data=[1,1,1,1,1,1,1,1,1],columns=col_names)
frames = []

second_run_wl_params = False
if second_run_wl_params == True:
    algorithm_list = ['QL ']
    problem_list = [ 'Stir Crazy ','C19 Grid ',]
    size_list = ['Small ', 'Medium ']#, 'Large ']
    gamma_list = ['G=0.7 ', 'G=0.8 ', 'G=0.9 ', 'G=0.99 ']
    epsilon_list = ['E=0.7 ','E=0.8 ','E=0.9 ', 'E=0.99 ']
    folder = 'run_stats/'
    f = '_run_stats_df.csv'
    key_metrics_folder = 'single_key_metric/'
    key_metrics_f = '_key_metrics_df.csv'
    col_names=['Time', 'Iteration', 'reward','row_name','algorithm','problem','size','gamma','epsilon']
    #concat_df = pd.DataFrame(data=[1,1,1,1,1,1,1,1,1],columns=col_names)
    frames = []




## QL CHARTS BY ITERATION (AND DF WILL ALL KEY METRICS)
give_ql_charts = False #True #
if give_ql_charts == True:
    for a in algorithm_list:
        for s in size_list:
            for g in gamma_list:
                for p in problem_list:
                    error_list = []
                    mean_v_list = []
                    max_v_list = []
                    alpha_list = []
                    time_list = []
                    e_list = []
                    for e in epsilon_list:
                        file_address = input_location+folder+a+p+s+g+e+f
                        df = pd.read_csv(file_address)
                        print(file_address)
                        #print(df)
                        Iteration = np.array(df['Iteration'])
                        Error = np.array(df['Error'])
                        Mean_V = np.array(df['Mean V'])
                        Max_V = np.array(df['Max V'])
                        Alpha = np.array(df['Alpha'])
                        Time = np.array(df['Time'])
                        error_list.append(Error)
                        mean_v_list.append(Mean_V)
                        max_v_list.append(Max_V)
                        alpha_list.append(Alpha)
                        time_list.append(Time)
                        e_list.append(e)

                        file_address2 = input_location+key_metrics_folder+a+p+s+g+e+key_metrics_f
                        df2 = pd.read_csv(file_address2)
                        row_name = a+p+s+g+e
                        df2['row_name'] = row_name
                        df2['algorithm'] = a
                        df2['problem'] = p
                        df2['size'] = s
                        df2['gamma'] = g
                        df2['epsilon'] = e
                        frames.append(df2)

                    error_array = np.vstack(error_list) #np.transpose()
                    mean_v_array = np.vstack(mean_v_list)
                    max_v_array = np.vstack(max_v_list)
                    alpha_array = np.vstack(alpha_list)
                    time_array = np.vstack(time_list)
                    print(error_array)
                    print("shape: ",error_array.shape)
                    chart_curve(Iteration, error_array, "Error: "+a+p+s+g,ql_chart_location\
                    , labels_list=e_list, xlabel="Iteration", ylabel="Error")
                    chart_curve(Iteration, mean_v_array, "Mean V: "+a+p+s+g,ql_chart_location\
                    , labels_list=e_list, xlabel="Iteration", ylabel="Mean V")
                    chart_curve(Iteration, max_v_array, "Max V: "+a+p+s+g,ql_chart_location\
                    , labels_list=e_list, xlabel="Iteration", ylabel="Max V")
                    chart_curve(Iteration, alpha_array, "Alpha: "+a+p+s+g,ql_chart_location\
                    , labels_list=e_list, xlabel="Iteration", ylabel="Alpha")
                    chart_curve(Iteration, time_array, "Time: "+a+p+s+g,ql_chart_location\
                    , labels_list=e_list, xlabel="Iteration", ylabel="Time")
                    print("Frames: \n",frames)
        df_key_metrics_ql = pd.concat(frames)
        df_key_metrics_ql.to_csv(ql_chart_location+"df_key_metrics_ql.csv")


'''NEXT CHART TONS OF CONVERGENCE CHARTS FOR PI AND VI
'''
algorithm_list = ['VI ','PI ']
problem_list = [ 'Stir Crazy ','C19 Grid ',]
size_list = ['Small ', 'Medium ', 'Large ']
gamma_list = ['G=0.8 ', 'G=0.9 ', 'G=0.95 ', 'G=0.99 ']
folder = 'run_stats/'
f = '_run_stats_df.csv'
key_metrics_folder = 'single_key_metric/'
key_metrics_f = '_key_metrics_df.csv'
col_names=['Time', 'Iteration', 'reward','row_name','algorithm','problem','size','gamma']
#concat_df = pd.DataFrame(data=[1,1,1,1,1,1,1,1,1],columns=col_names)
frames = []




## QL CHARTS BY ITERATION (AND DF WILL ALL KEY METRICS)
give_vi_pi_charts = True
if give_vi_pi_charts == True:
    for a in algorithm_list:
        for s in size_list:
            for p in problem_list:
                error_list = []
                mean_v_list = []
                max_v_list = []
                time_list = []
                g_list = []
                iter_list = []
                for g in gamma_list:
                    file_address = input_location+folder+a+p+s+g+f
                    df = pd.read_csv(file_address)
                    print(file_address)
                    #print(df)
                    Iteration = np.array(df['Iteration'])
                    Error = np.array(df['Error'])
                    Mean_V = np.array(df['Mean V'])
                    Max_V = np.array(df['Max V'])
                    Time = np.array(df['Time'])
                    error_list.append(Error)
                    mean_v_list.append(Mean_V)
                    max_v_list.append(Max_V)
                    time_list.append(Time)
                    g_list.append(g)
                    iter_list.append(list(Iteration))

                    file_address2 = input_location+key_metrics_folder+a+p+s+g+key_metrics_f
                    df2 = pd.read_csv(file_address2)
                    row_name = a+p+s+g
                    df2['row_name'] = row_name
                    df2['algorithm'] = a
                    df2['problem'] = p
                    df2['size'] = s
                    df2['gamma'] = g
                    frames.append(df2)
                Iteration = max((x) for x in iter_list)
                error_array = fill_nan_in_arrays(error_list) # np.vstack(error_list) #np.transpose()
                mean_v_array = fill_nan_in_arrays(mean_v_list) #np.vstack(mean_v_list)
                max_v_array = fill_nan_in_arrays(max_v_list) #np.vstack(max_v_list)
                time_array = fill_nan_in_arrays(time_list) #np.vstack(time_list)
                print(error_array)
                print("shape: ",error_array.shape)
                chart_curve(Iteration, error_array, "Error: "+a+p+s,_chart_location+a+"_charts/"\
                , labels_list=g_list, xlabel="Iteration", ylabel="Error")
                chart_curve(Iteration, mean_v_array, "Mean V: "+a+p+s,_chart_location+a+"_charts/"\
                , labels_list=g_list, xlabel="Iteration", ylabel="Mean V")
                chart_curve(Iteration, max_v_array, "Max V: "+a+p+s,_chart_location+a+"_charts/"\
                , labels_list=g_list, xlabel="Iteration", ylabel="Max V")
                chart_curve(Iteration, time_array, "Time: "+a+p+s,_chart_location+a+"_charts/"\
                , labels_list=g_list, xlabel="Iteration", ylabel="Time")
                print("Frames: \n",frames)
        df_key_metrics = pd.concat(frames)
        df_key_metrics.to_csv(_chart_location+a+"_charts/"+"df_key_metrics"+a+".csv")


'''FINALLY, MAKE REWARDS BAR CHARTS FOR QL, PI, and VI
'''

#Time	Iteration	reward	row_name	algorithm	problem	size	gamma	epsilon


algorithm_list = ['QL ']
problem_list = [ 'Stir Crazy ','C19 Grid ',]
size_list = ['Small ', 'Medium ']#, 'Large ']
gamma_list = ['G=0.7 ', 'G=0.8 ', 'G=0.90 ', 'G=0.99 ']
epsilon_list = ['E=0.7 ','E=0.8 ', 'E=0.90 ','E=0.99 ']

make_bar_charts = False
if make_bar_charts == True:
    df_location = ql_chart_location+'df_key_metrics_ql.csv'
    df = pd.read_csv(df_location)
    for a in algorithm_list:
        for p in problem_list:
            for s in size_list:
                for g in gamma_list:
                    print(a,p,s,g)
                    temp_df = df[(df['problem'] == p) & (df['size'] == s) & (df['algorithm'] == a)] #

                    print("temp_df.columns ", temp_df.columns)
                    print(temp_df['gamma'])
                    temp_df['label'] = temp_df['gamma'] + ";" + temp_df['epsilon']
                    #temp_df['label'] = temp_df.apply(lambda x: x['gamma']+";"+x['epsilon'], axis=0)
                    labels_list = list(temp_df['label'])
                    rewards = list(temp_df['reward'])
                    title = 'Reward: '+a+p+s
                    loc = ql_chart_location
                    chart_hbars(labels_list, rewards, title, loc)
                    print(labels_list, title)

    algorithm_list = ['VI ','PI ']
    problem_list = [ 'Stir Crazy ','C19 Grid ',]
    size_list = ['Small ', 'Medium ', 'Large ']
    gamma_list = ['G=0.8 ', 'G=0.9 ', 'G=0.95 ', 'G=0.99 ']

    for a in algorithm_list:
        df = pd.read_csv(_chart_location+a+"_charts/"+"df_key_metrics"+a+".csv")
        for p in problem_list:
            for s in size_list:
                temp_df = df[(df['problem'] == p) & (df['size'] == s) & (df['algorithm'] == a)] #
                labels_list = list(temp_df['gamma'])
                rewards = list(temp_df['reward'])
                title = 'Reward: '+a+p+s
                loc = _chart_location+a+"_charts/"
                chart_bars(labels_list, rewards, title, loc)
                print(labels_list, title)


#Show optimal policy
