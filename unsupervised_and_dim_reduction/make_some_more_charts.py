import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

def chart_curve(x_array, y_array, title, output_location, labels_list=0\
, use_f1=False, xlabel=None, ylabel=None):
    plt.figure(figsize=(6.4*.5,4.8*.5))

    num_lines = y_array.shape[0]
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

    plt.legend(fontsize=7)
    plt.savefig(output_location+title+'.png')
    plt.clf()
    return

def do_that_thing(df, title, ylabel, xlabel):
    k_array = np.array(df['k'])
    data_df = df.iloc[:,1:]
    data_array = np.transpose(np.array(data_df))
    labels = data_df.columns
    chart_curve(k_array,data_array, title, chart_location, labels_list=labels,ylabel=ylabel,xlabel=xlabel)
    return

input_location = '/data/'
chart_location = '/charts/'

df = pd.read_csv(input_location+"AMZ_inertia_data.csv")
title = "AMZ KM Silhouette By DR"
ylabel = "Silhouette"
xlabel = "Dim Reduction"
do_that_thing(df, title, ylabel, xlabel)

df = pd.read_csv(input_location+"PIMA_inertia_data.csv")
title = "PIMA KM Silhouette By DR"
ylabel = "Silhouette"
xlabel = "Dim Reduction"
do_that_thing(df, title, ylabel, xlabel)

df = pd.read_csv(input_location+"AMZ_score_data.csv")
title = "AMZ KM LL Score By DR"
ylabel = "LL"
xlabel = "Dim Reduction"
do_that_thing(df, title, ylabel, xlabel)
print("Finished.")

df = pd.read_csv(input_location+"PIMA_score_data.csv")
title = "PIMA KM LL Score By DR"
ylabel = "LL"
xlabel = "Dim Reduction"
do_that_thing(df, title, ylabel, xlabel)
print("Finished.")
