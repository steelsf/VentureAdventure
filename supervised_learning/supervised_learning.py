## Import everything I will need for data exploration, modeling, checking and graphing
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import call
import random
from sklearn import tree
from sklearn import neural_network
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def digest_file(full_input_location, target, verbose = False, separator = ","):
    df = pd.read_csv(full_input_location, sep=separator)
    if verbose:
        print(df.head())
        print(df.info())
    df[target+"2"] = df[target]
    df_group_target = df.groupby(target)[target+"2"].count().to_frame().reset_index()
    target_names_array = df_group_target.index.values
    target_1_count = df_group_target.loc[target_names_array[0], target+"2"]
    target_2_count = df_group_target.loc[target_names_array[1], target+"2"]
    print("Group ", target_names_array[0], ": ", target_1_count)
    print("Group ", target_names_array[1], ": ", target_2_count)
    print("Group ", target_names_array[0], " is ", "{:.0%}".format(target_1_count / (target_1_count + target_2_count)))
    df = df.drop(target+"2", axis=1)
    return df


def split_and_scale(df, target, train_size=0.25, random_state=12345):
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)
    scaler = StandardScaler()
    #fit scaler on train data only (to not add info about test data)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def train_and_evaluate(model, X_train, X_test, y_train, y_test, print_outputs=False):
    #fit model and time it
    start_time_fit = time.perf_counter()
    model.fit(X_train, y_train)
    end_time_fit = time.perf_counter()
    time_used_to_fit = end_time_fit - start_time_fit
    #accuracy for train
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_f1 = f1_score(y_train, train_predictions)
    #accuracy for test
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions)
    if print_outputs == True:
        print('Train Accuracy: {0:.2f}'.format(train_accuracy * 100.0))
        print('Train F1: {0:.2f}'.format(train_f1 * 100.0))
        print('Test Accuracy: {0:.2f}'.format(test_accuracy * 100.0))
        print('Test F1: {0:.2f}'.format(test_f1 * 100.0))
        print('Time: {0:.4f}'.format(time_used_to_fit * 100.0))
    return train_accuracy, test_accuracy, train_f1, test_f1, time_used_to_fit


def initialize_models():
    model_tree = tree.DecisionTreeClassifier(min_samples_split=5, random_state=12345) #
    model_nn = neural_network.MLPClassifier(random_state=12345)
    model_boost = AdaBoostClassifier(random_state=12345)
    model_SVM = svm.SVC(random_state=12345)
    model_knn = KNeighborsClassifier()
    list_of_models = [model_tree, model_nn, model_boost, model_SVM, model_knn]
    return list_of_models


def print_results_from_models(list_of_models, list_of_model_names,X_train, X_test, y_train, y_test\
,use_f1=False):
    array_of_accuracies = np.zeros(5)
    array_of_f1 = np.zeros(5)
    for i in range(len(list_of_models)):
        print("\nResults for {} model:".format(list_of_model_names[i]))
        tr_A, te_A, tr_f1, te_f1, time = train_and_evaluate(list_of_models[i], X_train, X_test, y_train, y_test\
        , print_outputs=True)
        array_of_accuracies[i] = te_A
        array_of_f1[i] = te_f1
    if use_f1 == False:
        accuracy_sum = np.sum(array_of_accuracies)
        for i in range(len(list_of_model_names)):
            relative_accuracy = array_of_accuracies[i] / accuracy_sum * 5
            print("Relative accuracy of {0:.2f} from {1:.4f} for ".format(relative_accuracy, array_of_accuracies[i]\
            ), list_of_model_names[i])
        #print(array_of_accuracies)
        return array_of_accuracies
    else:
        f1_sum = np.sum(array_of_f1)
        for i in range(len(list_of_model_names)):
            relative_f1 = array_of_f1[i] / f1_sum * 5
            print("Relative f1 score of {0:.2f} from {1:.4f} for ".format(relative_f1, array_of_f1[i]\
            ), list_of_model_names[i])
        #print(array_of_f1)
        return array_of_f1


def get_param_grids(random_state):
    tree_param_grid = {
        'criterion' : ['gini', 'entropy'],
        'max_depth' : [4,6,7,8,9,10,12,20],
        'min_samples_split' : [2,3,4,5,8,10,20],
        'min_samples_leaf' : [1,2,3,4,5],
        'random_state' : [random_state]
    }
    nn_param_grid = {
        'activation' : ["identity", "logistic", "tanh", "relu"],
        'alpha' : [ 0.0001, 0.0005, 0.001, 0.01, 0.1, 0.2],  #0.00001, 0.00005,
        'hidden_layer_sizes': [(5,),(10,),(20,),(50,),(100,),(10,10),(100,100)],
        'max_iter' : [100,200,500],
        'verbose' : [False],
        'random_state' : [random_state]
    }
    boost_param_grid = {
        'algorithm':['SAMME', 'SAMME.R'],
        'base_estimator':[tree.DecisionTreeClassifier(max_depth=1), tree.DecisionTreeClassifier(max_depth=2),
                          tree.DecisionTreeClassifier(max_depth=4), tree.DecisionTreeClassifier(max_depth=6),
                          tree.DecisionTreeClassifier(max_depth=10), tree.DecisionTreeClassifier(max_depth=20)],
        'n_estimators': range(20,121,10), #[20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        'learning_rate' : [0.4, 0.2, 0.1, 0.05, 0.01, 0.005],
        'random_state' : [random_state]
    }
    SVM_param_grid = {
        'C': [0.1, 1, 10, 20, 100, 1000],
        'gamma': [1, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001],
        'kernel': ['rbf', 'linear', 'poly'],
        'random_state' : [random_state]
    }
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9, 10, 11, 12, 13, 15],
        'weights' : ["uniform", "distance"]
    }
    list_of_model_param_sets = list([tree_param_grid, nn_param_grid, boost_param_grid, SVM_param_grid, knn_param_grid])
    return list_of_model_param_sets


def get_fine_param_grids_dataset_1(random_state):  #written after first round of analysis
    tree_param_grid = {
        'criterion' : ['gini','entropy'],  #], 'entropy'],
        'max_depth' : [7,8,9],  #6,7,4,6, ,8,9,10,12,20
        'min_samples_split' : [10,12,14],  #,3,4,5,8,10,20], 4,5,6,7,8,10,
        'min_samples_leaf' : [1,2], #[5,1,3,4],  #1,2,3,4,
        'random_state' : [random_state]
    }
    nn_param_grid = {
        'activation' : [ "tanh","relu"], #, "relu"],  #"identity", "logistic",
        'alpha' : [0.0001, 0.001], #, 0.01],  #0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.1, 0.01, , 0.2
        'hidden_layer_sizes': [(10,), (20,), (100,10)], #,(12,),(10,5), (15,), (15,5)],  #(5,),,(20,),(50,),(100,),(10,10),(100,100)
        'max_iter' : [500], #, 700, 1000],  #100,200,
        'verbose' : [False],
        'random_state' : [random_state]
    }
    boost_param_grid = {
        'algorithm':['SAMME', 'SAMME.R'],
        'base_estimator':[tree.DecisionTreeClassifier(max_depth=2),tree.DecisionTreeClassifier(max_depth=3)],
                        #tree.DecisionTreeClassifier(max_depth=4), tree.DecisionTreeClassifier(max_depth=6),
                        #tree.DecisionTreeClassifier(max_depth=10), tree.DecisionTreeClassifier(max_depth=20)],
        'n_estimators': [110],#[100, 105, 110, 115], #range(20,121,10),
        'learning_rate' : [0.1, 1.2],  #0.4, 0.2, , 0.8,  0.05, 0.01, 0.005
        'random_state' : [random_state]
    }
    SVM_param_grid = {
        'C': [2,1.5], #, 12, 8], #,  100, 1000],  #1, 10, 20,
        'gamma': [0.01], #, 0.012, 0.008],  #['scale', 'auto'],
        'kernel': ['rbf','poly'], #, 'linear', ],,  #['rbf', 'linear']  #,'poly',
        #'degree':[3], #,5],
        'random_state' : [random_state]
    }
    knn_param_grid = {
        'n_neighbors': [ 11, 12, 13,15,17,19,20],   #,13, 14 #3, 5, 7, 9, 10, 11,  13, 15
        'weights' : ["uniform", "distance"]
    }
    list_of_model_param_sets = list([tree_param_grid, nn_param_grid, boost_param_grid, SVM_param_grid, knn_param_grid])
    return list_of_model_param_sets


def get_fine_param_grids_dataset_2(random_state):  #written after first round of analysis
    tree_param_grid = {
        'criterion' : ['gini'],  #['gini', 'entropy'],
        'max_depth' : [30],  #,29,31], #40 #,100], # [4,6,8,12,20],
        'min_samples_split' : [2], #,3],  #[2,3,4,5],
        'min_samples_leaf' : [1],  #[1,2,3,4,5],
        'random_state' : [random_state]
    }
    nn_param_grid = {
        'activation' : [ "relu"], #"tanh", #["identity", "logistic", "tanh", "relu"],
        'alpha' : [0.001], #, 0.0015],  #, 0.002], #[0.001, 0.01, 0.1],  #0.00001, 0.00005, 0.0001, 0.0005,
        'hidden_layer_sizes': [(100,100,20)], #, (100,100,25)], #, (100,100,10, 10)],
        #, (100,100,10,10), (100,100,100,10)],  #[(10,),(100,),(10,10),(100,100)],
        'max_iter' : [1000],  #[500, 1000],  #[100,200,500],
        'verbose' : [False],
        'random_state' : [random_state]
    }
    boost_param_grid = {
        'algorithm':['SAMME.R'],  #['SAMME', 'SAMME.R'],
        'base_estimator':[tree.DecisionTreeClassifier(max_depth=6)],  #, tree.DecisionTreeClassifier(max_depth=7)],#=12
        'n_estimators': [130], #,135,125], #200], #range(20,121,10), #[20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        'learning_rate' : [0.05], #, 0.06],  #[0.1, 0.05, 0.01, 0.005],
        'random_state' : [random_state]
    }
    SVM_param_grid = {
        'C': [1.0],  #[0.1, 1, 10, 100, 1000],
        'gamma': ['scale'],  #['scale', 'auto'],#[1, 0.1, 0.01, 0.001],  #, 0.0001
        'kernel': ['poly'],  #['rbf', 'linear']  #,'poly',
        'degree':[5], #,6],  #,7,9],
        'random_state' : [random_state]
    }
    knn_param_grid = {
        'n_neighbors': [5],  #[3, 5, 7, 9, 11, 21],
        'weights' : [ "distance"]  #"uniform",
    }
    list_of_model_param_sets = list([tree_param_grid, nn_param_grid, boost_param_grid, SVM_param_grid, knn_param_grid])
    return list_of_model_param_sets


def get_final_models_dataset_1(random_state):
    model_tree = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=12,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=random_state, splitter='best')
    model_nn = neural_network.MLPClassifier(activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(10,), learning_rate='constant',
              learning_rate_init=0.001, max_iter=500, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=random_state, shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)
    model_boost = AdaBoostClassifier(algorithm='SAMME',
                   base_estimator=tree.DecisionTreeClassifier(class_weight=None, criterion='gini',
                                                         max_depth=2, max_features=None, max_leaf_nodes=None,
                                                         min_impurity_decrease=0.0, min_impurity_split=None,
                                                         min_samples_leaf=1,min_samples_split=2,
                                                         min_weight_fraction_leaf=0.0,presort=False,
                                                         random_state=random_state,splitter='best'),
                   learning_rate=0.075, n_estimators=110, random_state=random_state)
    model_SVM = svm.SVC(C=2, cache_size=200, class_weight=None, coef0=0.0,
                        decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
                        max_iter=-1, probability=False, random_state=random_state, shrinking=True,
                        tol=0.001, verbose=False)
    model_knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=13, p=2,
                     weights='uniform')
    list_of_models = [model_tree, model_nn, model_boost, model_SVM, model_knn]
    return list_of_models


def get_final_models_dataset_2(random_state):
    model_tree = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=30,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=random_state, splitter='best')
    model_nn = neural_network.MLPClassifier(activation='relu', alpha=0.001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(100, 100, 20), learning_rate='constant',
              learning_rate_init=0.001, max_iter=1000, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=random_state, shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)
    model_boost = AdaBoostClassifier(algorithm='SAMME.R',
                   base_estimator=tree.DecisionTreeClassifier(class_weight=None,
                                                         criterion='gini',
                                                         max_depth=6,
                                                         max_features=None,
                                                         max_leaf_nodes=None,
                                                         min_impurity_decrease=0.0,
                                                         min_impurity_split=None,
                                                         min_samples_leaf=1,
                                                         min_samples_split=2,
                                                         min_weight_fraction_leaf=0.0,
                                                         presort=False,
                                                         random_state=None,
                                                         splitter='best'),
                   learning_rate=0.05, n_estimators=130, random_state=random_state)
    model_SVM = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=5, gamma='scale', kernel='poly',
                max_iter=-1, probability=False, random_state=random_state, shrinking=True,
                tol=0.001, verbose=False)

    model_knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='distance')
    list_of_models = [model_tree, model_nn, model_boost, model_SVM, model_knn]
    return list_of_models


@ignore_warnings(category=ConvergenceWarning)
def grid_search_print(list_of_model_param_sets, list_of_model_names, list_of_models, X_train, y_train\
, output_location, use_f1=False, prefix='round_1_'):
    print("\nNow grid searching optimal parameters...")
    scoring='accuracy'
    if use_f1 == True:
        scoring='f1' #make_scorer(f1_score , average='micro')
    for i in range(len(list_of_model_param_sets)):
        print("--{}--".format(list_of_model_names[i]))
        grid = GridSearchCV(list_of_models[i], list_of_model_param_sets[i], refit=True,cv=5, scoring=scoring )
        grid.fit(X_train, y_train)
        best_param_df = pd.DataFrame(grid.cv_results_)[['mean_test_score','mean_fit_time','params','rank_test_score']]
        best_param_df.sort_values(by='rank_test_score', inplace=True)
        #best_param_df = best_param_df.loc[:9,:]
        print(best_param_df)
        print(grid.best_estimator_)
        best_param_df.to_csv(output_location+prefix+list_of_model_names[i]+"_best_params.csv")
    return


def get_learning_curve_inputs(model, df, target, step_size=100, random_state=12345, use_f1=False):
    nrows = df.shape[0]
    array_train_sizes = np.array(range(step_size,nrows,step_size))
    array_train_accuracy = np.zeros(len(array_train_sizes))
    array_test_accuracy = np.zeros(len(array_train_sizes))
    counter = 0
    for i in range(step_size,nrows,step_size):
        train_size = i
        X_train, X_test, y_train, y_test = split_and_scale(df, target, train_size=train_size, random_state=random_state)
        train_accuracy, test_accuracy, train_f1, test_f1, time = train_and_evaluate(model, X_train, X_test, y_train, y_test\
        , print_outputs=False)
        array_train_accuracy[counter] = train_accuracy
        array_test_accuracy[counter] = test_accuracy
        if use_f1==True:
            array_train_accuracy[counter] = train_f1
            array_test_accuracy[counter] = test_f1
        counter = counter + 1
    array_train_and_test = np.vstack((array_train_accuracy, array_test_accuracy))
    return array_train_sizes, array_train_and_test


def make_5_learning_curves(list_of_models, df, target, prefix, list_of_model_names, labels_list, \
chart_output_location, step_size=100, random_state=12345, use_f1=False,xlabel="Train Sample Size"):
    for i in range(len(list_of_models)):
        model = list_of_models[i]
        model_name = list_of_model_names[i]
        temp_title = prefix + " " + model_name + " Learning Curve"
        array_train_sizes, array_train_and_test = get_learning_curve_inputs\
        (model, df, target, step_size=step_size, random_state=random_state, use_f1=use_f1)

        array_results = np.vstack((array_train_sizes, array_train_and_test))
        df_results = pd.DataFrame(array_results.transpose(), columns = ['train_size', 'train_score', 'test_score'])
        df_results.to_csv(chart_output_location+"/"+temp_title+".csv")
        chart_curve(array_train_sizes, array_train_and_test, labels_list, temp_title, chart_output_location
        , use_f1=use_f1)
    return

def chart_curve(x_array, y_array, labels_list, title, output_location, use_f1=False, xlabel="Train Sample Size"):
    print("xlabel: ", xlabel)
    num_lines = y_array.shape[0]
    ylabel="Accuracy"
    if use_f1 == True:
        ylabel="F1 Score"
    print("{} lines comming up...".format(num_lines))
    for i in range(num_lines):
        plt.plot(x_array, y_array[i], label = labels_list[i])
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel,fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.savefig(output_location+title+'.png')
    plt.clf()
    return


def tree_depth1(INPUT, random_state=12345):
    model = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=INPUT,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=12,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=random_state, splitter='best')
    return model


def tree_criterion1(INPUT, random_state=12345):
    model = tree.DecisionTreeClassifier(class_weight=None, criterion=INPUT, max_depth=8,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=12,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=random_state, splitter='best')
    return model



def boosting_depth1(INPUT, random_state=12345):
    model = AdaBoostClassifier(algorithm='SAMME',
               base_estimator=tree.DecisionTreeClassifier(class_weight=None, criterion='gini',
               max_depth=INPUT, max_features=None, max_leaf_nodes=None,min_impurity_decrease=0.0,
               min_impurity_split=None,min_samples_leaf=1,min_samples_split=2,min_weight_fraction_leaf=0.0,
               presort=False,random_state=random_state,splitter='best'),
               learning_rate=0.075, n_estimators=110, random_state=random_state)
    return model


def boosting_learning_rate1(INPUT, random_state=12345):
    model = AdaBoostClassifier(algorithm='SAMME',
               base_estimator=tree.DecisionTreeClassifier(class_weight=None, criterion='gini',
               max_depth=2, max_features=None, max_leaf_nodes=None,min_impurity_decrease=0.0,
               min_impurity_split=None,min_samples_leaf=1,min_samples_split=2,min_weight_fraction_leaf=0.0,
               presort=False,random_state=random_state,splitter='best'),
               learning_rate=INPUT, n_estimators=110, random_state=random_state)
    return model


def SVM_c1(INPUT, random_state=12345):
    model = svm.SVC(C=INPUT, cache_size=200, class_weight=None, coef0=0.0,
                        decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
                        max_iter=-1, probability=False, random_state=random_state, shrinking=True,
                        tol=0.001, verbose=False)
    return model


def SVM_kernel1(INPUT, random_state=12345):
    model = svm.SVC(C=2, cache_size=200, class_weight=None, coef0=0.0,
                        decision_function_shape='ovr', degree=5, gamma=0.01, kernel=INPUT,
                        max_iter=-1, probability=False, random_state=random_state, shrinking=True,
                        tol=0.001, verbose=False)
    return model


def nn_layers1(INPUT, random_state=12345):
    model = neural_network.MLPClassifier(activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=INPUT, learning_rate='constant',
              learning_rate_init=0.001, max_iter=500, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=random_state, shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)
    return model


def nn_alpha1(INPUT, random_state=12345):
    model = neural_network.MLPClassifier(activation='tanh', alpha=INPUT, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(10,), learning_rate='constant',
              learning_rate_init=0.001, max_iter=500, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=random_state, shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)
    return model


def nn_activation1(INPUT, random_state=12345):
    model = neural_network.MLPClassifier(activation=INPUT, alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(10,), learning_rate='constant',
              learning_rate_init=0.001, max_iter=500, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=random_state, shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)
    return model


def knn_nn1(INPUT, random_state=12345):
    model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=INPUT, p=2,
                     weights='uniform')
    return model


def knn_weights1(INPUT, random_state=12345):
    model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=13, p=2,
                     weights=INPUT)
    return model


def tree_depth2(INPUT, random_state=12345):
    model = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=INPUT,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=random_state, splitter='best')
    return model


def tree_criterion2(INPUT, random_state=12345):
    model = tree.DecisionTreeClassifier(class_weight=None, criterion=INPUT, max_depth=30,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=random_state, splitter='best')
    return model


def boosting_depth2(INPUT, random_state=12345):
    model = AdaBoostClassifier(algorithm='SAMME.R',
                   base_estimator=tree.DecisionTreeClassifier(class_weight=None,
                   criterion='gini',max_depth=INPUT,max_features=None,max_leaf_nodes=None,
                   min_impurity_decrease=0.0,min_impurity_split=None,
                   min_samples_leaf=1,min_samples_split=2,min_weight_fraction_leaf=0.0,
                   presort=False,random_state=None,splitter='best'),
                   learning_rate=0.05, n_estimators=130, random_state=random_state)
    return model


def boosting_learning_rate2(INPUT, random_state=12345):
    model = AdaBoostClassifier(algorithm='SAMME.R',
                   base_estimator=tree.DecisionTreeClassifier(class_weight=None,
                   criterion='gini',max_depth=6,max_features=None,max_leaf_nodes=None,
                   min_impurity_decrease=0.0,min_impurity_split=None,
                   min_samples_leaf=1,min_samples_split=2,min_weight_fraction_leaf=0.0,
                   presort=False,random_state=None,splitter='best'),
                   learning_rate=INPUT, n_estimators=130, random_state=random_state)
    return model


def SVM_c2(INPUT, random_state=12345):
    model = svm.SVC(C=INPUT, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=5, gamma='scale', kernel='poly',
                max_iter=-1, probability=False, random_state=random_state, shrinking=True,
                tol=0.001, verbose=False)
    return model


def SVM_kernel2(INPUT, random_state=12345):
    model = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=5, gamma='scale', kernel=INPUT,
                max_iter=-1, probability=False, random_state=random_state, shrinking=True,
                tol=0.001, verbose=False)
    return model


def nn_layers2(INPUT, random_state=12345):
    model = neural_network.MLPClassifier(activation='relu', alpha=0.001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=INPUT, learning_rate='constant',
              learning_rate_init=0.001, max_iter=1000, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=random_state, shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)
    return model


def nn_alpha2(INPUT, random_state=12345):
    model = neural_network.MLPClassifier(activation='relu', alpha=INPUT, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(100, 100, 20), learning_rate='constant',
              learning_rate_init=0.001, max_iter=1000, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=random_state, shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)
    return model


def nn_activation2(INPUT, random_state=12345):
    model = neural_network.MLPClassifier(activation=INPUT, alpha=0.001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(100, 100, 20), learning_rate='constant',
              learning_rate_init=0.001, max_iter=1000, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=random_state, shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)
    return model


def knn_nn2(INPUT, random_state=12345):
    model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=INPUT, p=2,
                     weights='distance')
    return model


def knn_weights2(INPUT, random_state=12345):
    model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights=INPUT)
    return model


def complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, model_function,\
input_list, prefix, output_location, random_state=12345, use_f1=False):
    array_train_accuracy = np.zeros(len(input_list))
    array_test_accuracy = np.zeros(len(input_list))
    for i in range(len(input_list)):
        INPUT = input_list[i]
        model = model_function(INPUT, random_state=random_state)
        train_accuracy, test_accuracy, train_f1, test_f1, time = train_and_evaluate(model, X_train, X_test, y_train, y_test\
        , print_outputs=False)
        array_train_accuracy[i] = train_accuracy
        array_test_accuracy[i] = test_accuracy
        if use_f1==True:
            array_train_accuracy[i] = train_f1
            array_test_accuracy[i] = test_f1
    array_train_and_test = np.vstack((array_train_accuracy, array_test_accuracy))
    df_results = pd.DataFrame(array_train_and_test.transpose(), columns = ['train_score', 'test_score'])
    df_results[parameter_name] = input_list
    array_x = np.array(input_list)
    temp_title = prefix + " {}: {} Complexity".format(model_name, parameter_name)
    df_results.to_csv(output_location+"/"+temp_title+".csv")
    labels_list = ['train', 'test']
    chart_curve(array_x, array_train_and_test, labels_list, temp_title, output_location, xlabel=parameter_name,\
    use_f1=use_f1)
    return


def main():
    input_location = '/Users/vwy957/Documents/ML/data/'
    output_location = '/Users/vwy957/Documents/ML/outputs/'
    chart_output_location = '/Users/vwy957/Documents/ML/charts/'
    random_state = 12345
    use_f1 =  False #True #
    file_name = 'pima_diabetes.csv'
    target = "Outcome"
    prefix="Pima_Diabetes"
    df = digest_file(input_location + file_name, target)
    X_train, X_test, y_train, y_test = split_and_scale(df, target, train_size=0.25, random_state=random_state)
    list_of_model_names = ['Decision Tree', 'Neural Network', 'Boosting', 'Support Vector Machine', 'K Nearest Neighbors']
    list_of_models = initialize_models()
    print("\n**********Plain vanilla for ", prefix, "**********")
    arr_of_acc = print_results_from_models(list_of_models, list_of_model_names,X_train, X_test, y_train, y_test\
    ,use_f1=use_f1)
    list_of_final_models = get_final_models_dataset_1(random_state)
    print("\n**********Optimized models for ", prefix, "**********")
    arr_of_acc = print_results_from_models(list_of_final_models, list_of_model_names,X_train, X_test, y_train, y_test\
    ,use_f1=use_f1)
    #list_of_model_param_sets = get_param_grids(random_state)
    list_of_model_param_sets = get_fine_param_grids_dataset_1(random_state)


    ####################################################################################
    ########################## Can use this to grid search #############################
    ####################################################################################
    #grid_search_print(list_of_model_param_sets, list_of_model_names, list_of_models, X_train, y_train\
    #, output_location, use_f1=use_f1, prefix=prefix)

    list_of_final_models = get_final_models_dataset_1(random_state)
    arr_of_acc = print_results_from_models(list_of_final_models, list_of_model_names,X_train, X_test, y_train, y_test\
    ,use_f1=use_f1)
    temp_title = prefix + " " +  list_of_model_names[0] + " Learning Curve"
    labels_list = ['train', 'test']
    #make_5_learning_curves(list_of_final_models, df, target, prefix, list_of_model_names, labels_list, \
    #chart_output_location, step_size=100, random_state=12345, use_f1=use_f1,xlabel="Train Sample Size")

    ####################################################################################
    #### Will keep complexity curves commented out, but you can uncomment as needed ####
    ####################################################################################
    model_name = "Boost"
    parameter_name = "max_depth"
    input_list = [2,3,4,5,6,7,8,9,10]
    input_function = boosting_depth1
    #complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, input_function,\
    #input_list, prefix, output_location, random_state=12345, use_f1=use_f1)

    model_name = "Boost"
    parameter_name = "learning_rate"
    input_list = [.025,0.05,0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
    input_function = boosting_learning_rate1
    #complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, input_function,\
    #input_list, prefix, output_location, random_state=12345, use_f1=use_f1)

    model_name = "Tree"
    parameter_name = "max_depth"
    input_list = range(2,20)
    input_function = tree_depth1
    #complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, input_function,\
    #input_list, prefix, output_location, random_state=12345, use_f1=use_f1)

    model_name = "Tree"
    parameter_name = "criterion"
    input_list = ['gini', 'entropy']
    input_function = tree_criterion1
    #complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, input_function,\
    #input_list, prefix, output_location, random_state=12345, use_f1=use_f1)

    model_name = "SVM"
    parameter_name = "C"
    input_list = [0.5,1,2,3,4,5,6,7,8,9,10]
    input_function = SVM_c1
    #complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, input_function,\
    #input_list, prefix, output_location, random_state=12345, use_f1=use_f1)

    model_name = "SVM"
    parameter_name = "kernel"
    input_list = ['rbf', 'linear', 'poly']
    input_function = SVM_kernel1
    #complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, input_function,\
    #input_list, prefix, output_location, random_state=12345, use_f1=use_f1)

    model_name = "NN"
    parameter_name = "Layers"
    input_list = [(10,), (10,10), (10,10,10), (100,)]
    input_function = nn_layers1
    #have to comment out the call to create a chart in the complexity curve function to avoid errors
    #complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, input_function,\
    #input_list, prefix, output_location, random_state=12345, use_f1=use_f1)

    model_name = "NN"
    parameter_name = "Alpha"
    input_list = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    input_function = nn_alpha1
    #complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, input_function,\
    #input_list, prefix, output_location, random_state=12345, use_f1=use_f1)

    model_name = "NN"
    parameter_name = "Activation"
    input_list = ["identity", "logistic", "tanh", "relu"]
    input_function = nn_activation1
    #complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, input_function,\
    #input_list, prefix, output_location, random_state=12345, use_f1=use_f1)

    model_name = "KNN"
    parameter_name = "n_neighbors"
    input_list = [1, 3, 5, 7, 9, 10, 11, 12, 13, 15]
    input_function = knn_nn1
    #complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, input_function,\
    #input_list, prefix, output_location, random_state=12345, use_f1=use_f1)

    model_name = "KNN"
    parameter_name = "weight"
    input_list = ["uniform", "distance"]
    input_function = knn_weights1
    #complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, input_function,\
    #input_list, prefix, output_location, random_state=12345, use_f1=use_f1)

    ####################################################################################
    ####################################################################################
    ########################## Now again for the second dataset ########################
    ####################################################################################
    ####################################################################################

    use_f1 = True #False
    prefix="Amazon_Reviews"
    file_name = 'amazon_data3.csv'
    target = "low_score_ind"

    print("\n**********Plain vanilla for ", prefix, "**********")
    arr_of_acc = print_results_from_models(list_of_models, list_of_model_names,X_train, X_test, y_train, y_test\
    ,use_f1=use_f1)
    list_of_final_models = get_final_models_dataset_1(random_state)
    print("\n**********Optimized models for ", prefix, "**********")
    arr_of_acc = print_results_from_models(list_of_final_models, list_of_model_names,X_train, X_test, y_train, y_test\
    ,use_f1=use_f1)
    #list_of_model_param_sets = get_param_grids(random_state)
    list_of_model_param_sets = get_fine_param_grids_dataset_2(random_state)


    ####################################################################################
    ########################## Can use this to grid search #############################
    ####################################################################################
    #grid_search_print(list_of_model_param_sets, list_of_model_names, list_of_models, X_train, y_train\
    #, output_location, use_f1=use_f1, prefix=prefix)

    list_of_final_models = get_final_models_dataset_1(random_state)
    arr_of_acc = print_results_from_models(list_of_final_models, list_of_model_names,X_train, X_test, y_train, y_test\
    ,use_f1=use_f1)
    temp_title = prefix + " " +  list_of_model_names[0] + " Learning Curve"
    labels_list = ['train', 'test']
    #make_5_learning_curves(list_of_final_models, df, target, prefix, list_of_model_names, labels_list, \
    #chart_output_location, step_size=100, random_state=12345, use_f1=use_f1,xlabel="Train Sample Size")

    ####################################################################################
    #### Will keep complexity curves commented out, but you can uncomment as needed ####
    ####################################################################################

    model_name = "Tree"
    parameter_name = "max_depth"
    input_list = [5,10,20,30,35,40,45,50]
    input_function = tree_depth2
    #complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, input_function,\
    #input_list, prefix, output_location, random_state=12345, use_f1=use_f1)

    model_name = "Tree"
    parameter_name = "criterion"
    input_list = ['gini', 'entropy']
    input_function = tree_criterion2
    #complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, input_function,\
    #input_list, prefix, output_location, random_state=12345, use_f1=use_f1)

    model_name = "Boost"
    parameter_name = "max_depth"
    input_list = [2,3,4,5,6,7,8,9,10]
    input_function = boosting_depth2
    #complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, input_function,\
    #input_list, prefix, output_location, random_state=12345, use_f1=use_f1)

    model_name = "Boost"
    parameter_name = "learning_rate"
    input_list = [.025,0.05,0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
    input_function = boosting_learning_rate2
    #complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, input_function,\
    #input_list, prefix, output_location, random_state=12345, use_f1=use_f1)

    model_name = "SVM"
    parameter_name = "C"
    input_list = [0.5,1,1.5]
    input_function = SVM_c2
    #complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, input_function,\
    #input_list, prefix, output_location, random_state=12345, use_f1=use_f1)

    model_name = "SVM"
    parameter_name = "kernel"
    input_list = ['rbf', 'linear', 'poly']
    input_function = SVM_kernel2
    #complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, input_function,\
    #input_list, prefix, output_location, random_state=12345, use_f1=use_f1)

    model_name = "NN"
    parameter_name = "Layers"
    input_list = [(10,), (10,10), (10,10,10), (100,)]
    input_function = nn_layers2
    #have to comment out the call to create a chart in the complexity curve function to avoid errors
    #complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, input_function,\
    #input_list, prefix, output_location, random_state=12345, use_f1=use_f1)

    model_name = "NN"
    parameter_name = "Alpha"
    input_list = [0.00001,  0.001,  0.1]
    input_function = nn_alpha2
    #complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, input_function,\
    #input_list, prefix, output_location, random_state=12345, use_f1=use_f1)

    model_name = "NN"
    parameter_name = "Activation"
    input_list = ["identity", "logistic", "tanh", "relu"]
    input_function = nn_activation2
    complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, input_function,\
    input_list, prefix, output_location, random_state=12345, use_f1=use_f1)

    model_name = "KNN"
    parameter_name = "n_neighbors"
    input_list = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    input_function = knn_nn2
    #complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, input_function,\
    #input_list, prefix, output_location, random_state=12345, use_f1=use_f1)

    model_name = "KNN"
    parameter_name = "weight"
    input_list = ["uniform", "distance"]
    input_function = knn_weights2
    #complexity_curve(X_train, X_test, y_train, y_test, model_name, parameter_name, input_function,\
    #input_list, prefix, output_location, random_state=12345, use_f1=use_f1)


if __name__ == "__main__":
    main()



'''



PIMA for F1
def get_fine_param_grids_dataset_1(random_state):  #written after first round of analysis
    tree_param_grid = {
        'criterion' : ['gini','entropy'],  #], 'entropy'],
        'max_depth' : [6,7,8],  #4,6, ,8,9,10,12,20
        'min_samples_split' : [8,10,12],  #,3,4,5,8,10,20],
        'min_samples_leaf' : [5,1,3,4],  #1,2,3,4,
        'random_state' : [random_state]
    }
    nn_param_grid = {
        'activation' : [ "tanh", "relu"],  #"identity", "logistic",
        'alpha' : [0.0001, 0.01],  #0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.1, 0.01, , 0.2
        'hidden_layer_sizes': [(10,),(12,),(10,5), (15,), (15,5)],  #(5,),,(20,),(50,),(100,),(10,10),(100,100)
        'max_iter' : [50, 100, 150],  #100,200,
        'verbose' : [False],
        'random_state' : [random_state]
    }
    boost_param_grid = {
        'algorithm':['SAMME', 'SAMME.R'],
        'base_estimator':[tree.DecisionTreeClassifier(max_depth=3), tree.DecisionTreeClassifier(max_depth=2)
                         ,tree.DecisionTreeClassifier(max_depth=4),tree.DecisionTreeClassifier(max_depth=7)],
        #tree.DecisionTreeClassifier(max_depth=4), tree.DecisionTreeClassifier(max_depth=6),
        #tree.DecisionTreeClassifier(max_depth=10), tree.DecisionTreeClassifier(max_depth=20)],
        'n_estimators': [100, 105, 110, 115], #range(20,121,10), #[20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        'learning_rate' : [ 0.1, 1.5, 2.0],  #0.4, 0.2, , 0.05, 0.01, 0.005
        'random_state' : [random_state]
    }

'''
