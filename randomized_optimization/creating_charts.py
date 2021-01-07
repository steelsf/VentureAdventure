import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_csv_drop_zero_NN(location):
    temp_df = pd.read_csv(location)
    temp_df = temp_df[(temp_df['iterations'] > 0)].copy()
    return temp_df

def read_csv_drop_zero(location):
    temp_df = pd.read_csv(location)
    temp_df = temp_df[(temp_df['max_iter'] > 0)].copy()
    return temp_df

def accuracy_and_time(location):
    temp_df = read_csv_drop_zero_NN(location)
    accuracy_array = np.array(temp_df['accuracy'])
    time_array = np.array(temp_df['avg_time'])
    return accuracy_array, time_array

def fit_and_time(location):
    temp_df = read_csv_drop_zero(location)
    row_num = temp_df.shape[0]
    fit_array = np.array(temp_df['avg_fit'])
    time_array = np.array(temp_df['avg_time'])
    if row_num < 10:
        temp_array1 = np.array([np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN])
        temp_array2 = np.array([np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN])
        temp_array1[0:row_num] = fit_array
        fit_array = temp_array1
        temp_array2[0:row_num] = time_array
        time_array = temp_array2
    return fit_array, time_array

def fit_and_time_from_curve(location):
    temp_df = pd.read_csv(location)
    temp_array1 = temp_df.iloc[:,0]
    temp_len = len(temp_array1)
    temp_array2 = np.array([np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN,np.NaN])
    temp_array2[0:temp_len] = temp_array1
    return temp_array2

def chart_curve(x_array, y_array, labels_list, title, output_location, ylabel="Fit", xlabel="Iterations"):
    print("xlabel: ", xlabel)
    num_lines = y_array.shape[0]
    print("this is num lines: ", num_lines)
    ylabel=ylabel
    print("{} lines comming up...".format(num_lines))
    for i in range(num_lines):
        plt.plot(x_array, y_array[i], label = labels_list[i])
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel,fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid(False)
    plt.legend(fontsize=14)
    plt.savefig(output_location+title+'.jpeg')
    plt.clf()
    return

input_location = 'outputs/'
chart_location = 'charts/'
name_dict = {
'GD_NN':'nn_GD_learning2.csv',
'GA_NN':'nn_GA_learning2.csv',
'GA_4P':'final_GA_4P_attempt_8am.csv',
'GA_FF':'final_GA_FF_attempt_8am.csv',
'GA_KS':'final_GA_KS_attempt_8am.csv',
'SA_NN':'nn_SA_learning2.csv',
'SA_4P':'final_SA_4P_attempt_8am.csv',
'SA_FF':'final_SA_FF_attempt_8am.csv',
'SA_KS':'final_SA_KS_attempt_8am.csv',
'RHC_NN':'nn_RHC_learning2.csv',
'RHC_4P':'final_RHS_4P_attempt_8am.csv',
'RHC_FF':'final_RHS_FF_attempt_8am.csv',
'RHC_KS':'final_RHS_KS_attempt_8am.csv',
'MIMIC_4P':'final_MIMIC_4P_attempt_3am.csv',
'MIMIC_FF':'final_MIMIC_FF_attempt_3am.csv',
'MIMIC_KS':'final_MIMIC_KS_attempt_3am.csv',


'GA_4P_big':'final_GA_4P_big_attempt_8am.csv',
'GA_4P_small':'final_GA_4P_small_attempt_8am.csv',
'GA_FF_big':'final_GA_FF_big_attempt_8am.csv',
'GA_FF_small':'final_GA_FF_small_attempt_8am.csv',
'GA_KS_big':'final_GA_KS_big_attempt_8am.csv',
'GA_KS_small':'final_GA_KS_small_attempt_8am.csv',

'SA_4P_big':'final_SA_4P_big_attempt_8am.csv',
'SA_4P_small':'final_SA_4P_small_attempt_8am.csv',
'SA_FF_big':'final_SA_FF_big_attempt_8am.csv',
'SA_FF_small':'final_SA_FF_small_attempt_8am.csv',
'SA_KS_big':'final_SA_KS_big_attempt_8am.csv',
'SA_KS_small':'final_SA_KS_small_attempt_8am.csv',

'RHC_4P_big':'final_RHS_4P_big_attempt_8am.csv',
'RHC_4P_small':'final_RHS_4P_small_attempt_8am.csv',
'RHC_FF_big':'final_RHS_FF_big_attempt_8am.csv',
'RHC_FF_small':'final_RHS_FF_small_attempt_8am.csv',
'RHC_KS_big':'final_RHS_KS_big_attempt_8am.csv',
'RHC_KS_small':'final_RHS_KS_small_attempt_8am.csv',

'MIMIC_4P_big':'MIMIC_4P_big_short_curve.csv',
'MIMIC_4P_small':'MIMIC_4P_small_short_curve.csv',
'MIMICFF_big':'MIMIC_FF_big_full_curve.csv',
'MIMIC_FF_small':'MIMIC_FF_small_full_curve.csv',
'MIMIC_KS_big':'MIMIC_KS_big_short_curve.csv',
'MIMIC_KS_small':'MIMIC_KS_big_short_curve.csv',


'GA_KS_mutation_01':'final_GA_KS_mutation_01.csv',
'GA_KS_pop_100':'final_GA_KS_mutation_02_pop_100.csv',
'SA_4P_decay_99':'final_SA_4P_decay_99.csv',
'SA_4P_T_1':'final_SA_4P_T_1_decay_80.csv',
'MIMIC_FF_p_100_k_20':'MIMIC_FF_p_100_k_20_short_curve.csv',
'MIMIC_FF_p_150_k_20':'MIMIC_FF_p_150_k_20_short_curve.csv',
'MIMIC_FF_p_150_k_50':'MIMIC_FF_p_150_k_50_short_curve.csv'


}

iters = np.array([2,4,8,16,32,64,128,256,512,1024])
print("shape of iters: ", iters.shape)

RHC_NN_accuracy, RHC_NN_time = accuracy_and_time(input_location + name_dict['RHC_NN'])
SA_NN_accuracy, SA_NN_time = accuracy_and_time(input_location + name_dict['SA_NN'])
GA_NN_accuracy, GA_NN_time = accuracy_and_time(input_location + name_dict['GA_NN'])
GD_NN_accuracy, GD_NN_time = accuracy_and_time(input_location + name_dict['GD_NN'])

RHC_4P_fit, RHC_4P_time = fit_and_time(input_location + name_dict['RHC_4P'])
SA_4P_fit, SA_4P_time = fit_and_time(input_location + name_dict['SA_4P'])
GA_4P_fit, GA_4P_time = fit_and_time(input_location + name_dict['GA_4P'])
MIMIC_4P_fit, MIMIC_4P_time = fit_and_time(input_location + name_dict['MIMIC_4P'])

RHC_KS_fit, RHC_KS_time = fit_and_time(input_location + name_dict['RHC_KS'])
SA_KS_fit, SA_KS_time = fit_and_time(input_location + name_dict['SA_KS'])
GA_KS_fit, GA_KS_time = fit_and_time(input_location + name_dict['GA_KS'])
MIMIC_KS_fit, MIMIC_KS_time = fit_and_time(input_location + name_dict['MIMIC_KS'])

RHC_FF_fit, RHC_FF_time = fit_and_time(input_location + name_dict['RHC_FF'])
SA_FF_fit, SA_FF_time = fit_and_time(input_location + name_dict['SA_FF'])
GA_FF_fit, GA_FF_time = fit_and_time(input_location + name_dict['GA_FF'])
MIMIC_FF_fit, MIMIC_FF_time = fit_and_time(input_location + name_dict['MIMIC_FF'])


GA_KS_mut_01_fit, GA_KS_mut_01_time = fit_and_time(input_location + name_dict['GA_KS_mutation_01'])
GA_KS_pop_100_fit, GA_KS_pop_100_time = fit_and_time(input_location + name_dict['GA_KS_pop_100'])
SA_4P_decay_99_fit, SA_4P_decay_99_time = fit_and_time(input_location + name_dict['SA_4P_decay_99'])
SA_4P_T_1_fit, SA_4P_T_1_time = fit_and_time(input_location + name_dict['SA_4P_T_1'])
MIMIC_FF_p_100_k_20_fit = fit_and_time_from_curve(input_location + name_dict['MIMIC_FF_p_100_k_20'])
MIMIC_FF_p_150_k_20_fit = fit_and_time_from_curve(input_location + name_dict['MIMIC_FF_p_150_k_20'])
MIMIC_FF_p_150_k_50_fit = fit_and_time_from_curve(input_location + name_dict['MIMIC_FF_p_150_k_50'])



print("Now I start to make vstacks and charts")
NN_accuracies = np.vstack((RHC_NN_accuracy, SA_NN_accuracy, GA_NN_accuracy, GD_NN_accuracy))
chart_curve(iters, NN_accuracies, ['RCH','SA','GA','GD'], "NN Fit", chart_location, ylabel="Fit", xlabel="Iterations")
NN_time = np.vstack((RHC_NN_time, SA_NN_time, GA_NN_time, GD_NN_time))
chart_curve(iters, NN_time, ['RCH','SA','GA','GD'], "NN Time", chart_location, ylabel="Time", xlabel="Iterations")

P4_fit = np.vstack((RHC_4P_fit, SA_4P_fit, GA_4P_fit, MIMIC_4P_fit))
chart_curve(iters, P4_fit, ['RCH','SA','GA','MIMIC'], "4P Fit", chart_location, ylabel="Fit", xlabel="Iterations")
P4_time = np.vstack((RHC_4P_time, SA_4P_time, GA_4P_time, MIMIC_4P_time))
chart_curve(iters, P4_time, ['RCH','SA','GA','MIMIC'], "4P Time", chart_location, ylabel="Time", xlabel="Iterations")

FF_fit = np.vstack((RHC_FF_fit, SA_FF_fit, GA_FF_fit, MIMIC_FF_fit))
chart_curve(iters, FF_fit, ['RCH','SA','GA','MIMIC'], "FF Fit", chart_location, ylabel="Fit", xlabel="Iterations")
FF_time = np.vstack((RHC_FF_time, SA_FF_time, GA_FF_time, MIMIC_FF_time))
chart_curve(iters, FF_time, ['RCH','SA','GA','MIMIC'], "FF Time", chart_location, ylabel="Time", xlabel="Iterations")

KS_fit = np.vstack((RHC_KS_fit, SA_KS_fit, GA_KS_fit, MIMIC_KS_fit))
chart_curve(iters, KS_fit, ['RCH','SA','GA','MIMIC'], "KS Fit", chart_location, ylabel="Fit", xlabel="Iterations")
KS_time = np.vstack((RHC_KS_time, SA_KS_time, GA_KS_time, MIMIC_KS_time))
chart_curve(iters, KS_time, ['RCH','SA','GA','MIMIC'], "KS Time", chart_location, ylabel="Time", xlabel="Iterations")

GA_KS_fit_grid = np.vstack((GA_KS_mut_01_fit, GA_KS_pop_100_fit, GA_KS_fit))
chart_curve(iters, GA_KS_fit_grid, ['Mut=0.1; Pop=200','Mut=0.2; Pop=100','Mut=0.2; Pop=200'], "GA KS Fit By Params", chart_location)
GA_KS_time_grid = np.vstack((GA_KS_mut_01_time, GA_KS_pop_100_time, GA_KS_time))
chart_curve(iters, GA_KS_time_grid, ['Mut=0.1; Pop=200','Mut=0.2; Pop=100','Mut=0.2; Pop=200'], "GA KS Time By Params", chart_location)

SA_4P_fit_grid = np.vstack((SA_4P_decay_99_fit, SA_4P_T_1_fit, SA_4P_fit))
chart_curve(iters, SA_4P_fit_grid, ['D=0.99;T=100','D=0.8;T=1','D=0.8;T=100'], "SA 4P Fit By Params", chart_location)
SA_4P_time_grid = np.vstack((SA_4P_decay_99_time, SA_4P_T_1_time, SA_4P_time))
chart_curve(iters, SA_4P_time_grid, ['D=0.99;T=100','D=0.8;T=1','D=0.8;T=100'], "SA 4P Time By Params", chart_location)

MIMIC_FF_fit_grid = np.vstack((MIMIC_FF_p_100_k_20_fit, MIMIC_FF_p_150_k_50_fit, MIMIC_FF_fit))
chart_curve(iters, SA_4P_fit_grid, ['P=100;K=0.2','P=150;K=0.5','P=100;K=0.2'], "MIMIC FF Fit By Params", chart_location)

MIMIC_FF_p_150_k_50_fit
#'GA_KS_mutation_01':'final_GA_KS_mutation_01.csv',
#'GA_KS_pop_100':'final_GA_KS_mutation_02_pop_100.csv',




print("Finished.")
