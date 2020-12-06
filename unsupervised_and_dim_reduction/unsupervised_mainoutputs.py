## Import everything I will need for data exploration, modeling, checking and graphing
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import random
import pickle
from statistics import mean
#import seaborn as sns
from sklearn.pipeline import Pipeline
from scipy.stats import kurtosis
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn import neural_network
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import DataDimensionalityWarning
from sklearn.svm import SVR

from yellowbrick.cluster import InterclusterDistance
from yellowbrick.cluster import SilhouetteVisualizer


def split_and_scale(df, target, train_size=0.75, random_state=12345):
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)
    scaler = StandardScaler()
    #fit scaler on train data only (to not add info about test data)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def kmeans_by_k(data, output_location, prefix="", title="default", k_list=range(1,21,1),  n_clusters=20, init='k-means++'\
, n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True,  algorithm='auto'):
    print("Creating KMeans df and charts for ", title)
    inertia_list = []
    iterations_list = []
    silhouette_list = []
    time_list = []
    rand = random_state
    for k in k_list:
        for r in range(5):
            pre_inertia_list = []
            pre_iterations_list = []
            pre_silhouette_list = []
            pre_time_list = []
            random_state = rand * r
            kmeans = KMeans(n_clusters=k, init=init, max_iter=max_iter, tol=tol
            , verbose=verbose, random_state=random_state, copy_x=copy_x,  algorithm=algorithm)
            start_time_fit = time.perf_counter()
            kmeans.fit(data)
            end_time_fit = time.perf_counter()
            time_used_to_fit = end_time_fit - start_time_fit
            print("inertia: ",kmeans.inertia_)
            pre_inertia_list.append(kmeans.inertia_)
            pre_iterations_list.append(kmeans.n_iter_)
            pre_time_list.append(time_used_to_fit)
            cluster_labels = kmeans.predict(data)
            if k == 1:
                pre_silhouette_list.append(np.NaN)
            else:
                pre_silhouette_list.append(silhouette_score(data, cluster_labels))
        inertia_list.append(mean(pre_inertia_list))
        iterations_list.append(mean(pre_iterations_list))
        if k == 1:
            silhouette_list.append(np.NaN)
        else:
            silhouette_list.append(mean(pre_silhouette_list))
        time_list.append(mean(pre_time_list))
    inertia_df = pd.DataFrame({'k':k_list,'inertia':inertia_list,'iterations':iterations_list\
    ,'silhouette':silhouette_list, 'time':time_list})
    inertia_df.to_csv(output_location+prefix+'_inertia_'+title+'.csv', index=False)
    inertia_array = np.array(inertia_list)
    k_array = np.array(k_list)
    silhouette_array = np.array(silhouette_list)
    time_array = np.array(time_list)
    chart_curve(k_array, inertia_array, prefix+' KMeans SSE '+title, output_location\
    ,xlabel='Clusters', ylabel='SSE')
    chart_curve(k_array, silhouette_array, prefix + ' KMeans Silhouette Score '+title, output_location\
    ,xlabel='Clusters', ylabel='Silhouette')
    chart_curve(k_array, time_array, prefix + ' Time to Fit KMeans Clusters '+title, output_location\
    ,xlabel='Clusters', ylabel='Seconds')
    return inertia_df


def EM_by_k(data, output_location, prefix="",title="default", k_list=range(1,21,1), covariance_type='full', tol=0.001\
, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None\
, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10):
    print("Creating EM df and charts for ",title)
    score_list = []
    iterations_list = []
    aic_list = []
    bic_list = []
    time_list = []
    rand = random_state
    print("type(rand)",type(rand))
    for k in k_list:
        pre_score_list = []
        pre_iterations_list = []
        pre_aic_list = []
        pre_bic_list = []
        pre_time_list = []
        for r in range(5):
            random_state = rand * r
            EMax = GaussianMixture(n_components=k, covariance_type=covariance_type, tol=tol\
            , reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params\
            , weights_init=weights_init, means_init=means_init, precisions_init=precisions_init\
            , random_state=random_state, warm_start=warm_start, verbose=verbose, verbose_interval=verbose_interval)
            start_time_fit = time.perf_counter()
            EMax.fit(data)
            end_time_fit = time.perf_counter()
            time_used_to_fit = end_time_fit - start_time_fit
            print("score: ",EMax.score(data))
            pre_score_list.append(EMax.score(data))
            pre_iterations_list.append(EMax.n_iter_)
            pre_aic_list.append(EMax.aic(data))
            pre_bic_list.append(EMax.bic(data))
            pre_time_list.append(time_used_to_fit)
        score_list.append(mean(pre_score_list))
        iterations_list.append(mean(pre_iterations_list))
        aic_list.append(mean(pre_aic_list))
        bic_list.append(mean(pre_bic_list))
        time_list.append(mean(pre_time_list))
    score_df = pd.DataFrame({'k':k_list,'score':score_list,'iterations':iterations_list,'aic':aic_list\
    ,'bic':bic_list})
    score_df.to_csv(output_location+prefix+'_score_'+title+'.csv', index=False)
    score_array = np.array(score_list)
    k_array = np.array(k_list)
    aic_array = np.array(aic_list)
    bic_array = np.array(bic_list)
    time_array = np.array(time_list)
    chart_curve(k_array, score_array, prefix+' EM Log-Likelihood '+title, output_location\
    ,xlabel='Clusters', ylabel='LL')
    chart_curve(k_array, aic_array, prefix+' AIC '+title, output_location\
    ,xlabel='Clusters', ylabel='AIC')
    chart_curve(k_array, bic_array, prefix+' BIC '+title, output_location\
    ,xlabel='Clusters', ylabel='BIC')
    chart_curve(k_array, time_array, prefix+' Time to Fit EM Clusters '+title, output_location\
    ,xlabel='Clusters', ylabel='Seconds')
    return score_df


@ignore_warnings(category=DataDimensionalityWarning)
def rand_proj_evaluation(data, output_location, prefix="",random_state=1234, title='', k_list = list(range(1,12,1))):
    list_of_dist_list = []
    list_of_time_list = []
    for r in range(10):
        dist_list = []
        time_list = []
        r = r * random_state
        for i in k_list:
            rand_proj = GaussianRandomProjection(n_components=i,  eps=0.1, random_state=r)
            start_time_fit = time.perf_counter()
            data_transformed = rand_proj.fit_transform(data)
            end_time_fit = time.perf_counter()
            time_used_to_fit = end_time_fit - start_time_fit
            components = rand_proj.components_
            inverse_components = np.linalg.pinv(rand_proj.components_)
            data_transformed_back = data_transformed @ inverse_components.transpose()
            dist = np.linalg.norm(data-data_transformed_back)
            dist_list.append(dist)
            time_list.append(time_used_to_fit)
        list_of_dist_list.append(dist_list)
        list_of_time_list.append(time_list)
    temp_df = pd.DataFrame({'k':k_list, 'dist':dist_list, 'time':time_list})
    k_array = np.array(k_list)
    print("k_array: ",k_array.shape)
    dist_array = np.array(list_of_dist_list)
    print(dist_array.shape,"\n\n\n\n\n\n")
    time_array = np.array(list_of_time_list)
    chart_curve(k_array, dist_array, prefix+' Rand. Proj. Reconstrution Error' + title, output_location\
    ,labels_list=list(range(dist_array.shape[1])),ylabel='Reconstruction Error', xlabel='# of Projections')
    chart_curve(k_array, time_array, prefix+' Rand. Proj. Fit Time' + title, output_location\
    ,labels_list=list(range(time_array.shape[1])),ylabel='Time to Fit', xlabel='# of Projections')
    return


def create_PCA_charts(pca, output_location, title="",prefix=""):
    x_axis = list(range(1,pca.n_components_+1,1))
    chart_bars(x_axis, pca.singular_values_, prefix+' PCA Eigen Values '+title, output_location\
    ,xlabel='Principle Components', ylabel='Value')
    chart_bars(x_axis, pca.explained_variance_ratio_, prefix+' PCA Explained Variace Ratio '+title, output_location\
    ,xlabel='Principle Components', ylabel='% of Variance')
    return


def pca_vs_rand_reconstruction(data, output_location, random_state=12345, title="",prefix=""):
    pca = PCA(random_state=random_state)
    pca.fit(data)
    components_list = list(range(1,len(pca.components_)))

    pca_error = []
    rp_error = []
    for i in components_list:
        #pca
        mod = PCA(n_components=i, random_state=random_state)
        data_transformed = mod.fit_transform(data)
        components = mod.components_
        inverse_components = np.linalg.pinv(components)
        data_transformed_back = data_transformed @ inverse_components.transpose()
        dist_error = np.linalg.norm(data-data_transformed_back)
        dist = np.linalg.norm(data)
        error = dist_error / dist
        pca_error.append(error)
        #rp
        mod = GaussianRandomProjection(n_components=i, random_state=random_state)
        data_transformed = mod.fit_transform(data)
        components = mod.components_
        inverse_components = np.linalg.pinv(components)
        data_transformed_back = data_transformed @ inverse_components.transpose()
        dist_error = np.linalg.norm(data-data_transformed_back)
        dist = np.linalg.norm(data)
        error = dist_error / dist
        rp_error.append(error)
    components_array = np.array(components_list)
    pca_error_array = np.array(pca_error)
    rp_error_array = np.array(rp_error)
    error_array = np. vstack([pca_error_array,rp_error_array])
    label_list = ['pca','rp']
    chart_curve(components_array, error_array, prefix+' % Reconstruction Error ' + title, output_location\
    ,labels_list=label_list,ylabel='Reconstruction Error %', xlabel='# of Compoments')

    return


@ignore_warnings(category=ConvergenceWarning)
def grid_search_print(model_params, title, model, X_train, y_train\
, output_location, use_f1=False, prefix='round_1_'):
    print("\nNow grid searching optimal parameters...")
    scoring='accuracy'
    if use_f1 == True:
        scoring='f1' #make_scorer(f1_score , average='micro')

    print("--{}--".format(title))
    grid = GridSearchCV(model, model_params, refit=True,cv=3, scoring=scoring )
    grid.fit(X_train, y_train)
    best_param_df = pd.DataFrame(grid.cv_results_)[['mean_test_score','mean_fit_time','params','rank_test_score']]
    best_param_df.sort_values(by='rank_test_score', inplace=True)
    print(best_param_df)
    print(grid.best_estimator_)
    best_param_df.to_csv(output_location+prefix+title+"_best_params.csv")
    return


def analyze_clusters(clusterer, data, y_data, output_location, model='kmeans', prefix="", title=""):
    clusterer.fit(data)
    if model == 'kmeans':
        clusters = clusterer.cluster_centers_
        score_or_inertia = clusterer.inertia_
        plt.figure(figsize=(6.4,4.8))
        visualizer = InterclusterDistance(clusterer)
        visualizer.fit(data)        # Fit the data to the visualizer
        title1 = prefix+" "+title+" Inter Cluster Distance Map"
        plt.title(title1, fontsize="24")
        plt.savefig(output_location+title1+'.png')
        plt.clf()
        plt.close('all')
        plt.figure(figsize=(6.4,4.8))
        visualizer = SilhouetteVisualizer(clusterer, colors='yellowbrick')
        visualizer.fit(data)        # Fit the data to the visualizer
        title2 = prefix+" "+title+" Silhouette"

        plt.title(title2, fontsize="24")
        plt.savefig(output_location+title2+'.png')
        plt.clf()
        plt.close('all')
    else:
        score_or_inertia = clusterer.score(data)
        log_likely_dist = clusterer.score_samples(data)
        title2 = prefix+" "+title+" Hist of Log Likelihood"
        plt.close('all')
        plt.figure(figsize=(6.4*.5,4.8*.5)) #steel_note
        plt.hist(log_likely_dist, bins = 20)
        plt.title(title2, fontsize=10)
        plt.grid(False)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.savefig(output_location+title2+'.png')
        plt.clf()
        plt.close('all')
        clusters = clusterer.means_
    num_clusters = len(clusters)
    dists = euclidean_distances(clusters)
    tri_dists = dists[np.triu_indices(num_clusters,1)]
    max_dist, avg_dist, min_dist = tri_dists.max(), tri_dists.mean(), tri_dists.min()
    temp_df = pd.DataFrame({'score_or_inertia':[score_or_inertia],'avg_inter_dist':[avg_dist],'max_inter_dist':[max_dist]\
    ,'min_inter_dist':[min_dist],'model':model})
    temp_df.to_csv(output_location+prefix+" "+title+'_cluster_metrics.csv')
    cluster_pred = clusterer.predict(data)
    combined_x_y = pd.DataFrame({'y':y_data, 'cluster':cluster_pred})
    list_of_perc_1 = []
    list_of_perc_contribution_to_1 = []
    list_of_perc_contribution_to_0 = []
    list_of_total_in_cluster = []
    list_of_clusters = list(range(num_clusters))
    nrows = len(combined_x_y.index)
    n_in_1 = len(combined_x_y[(combined_x_y['y']==1)].index)
    n_in_0 = nrows = n_in_1
    for i in list_of_clusters:
        temp_df2 = combined_x_y[(combined_x_y['cluster']==i)]
        cluster_total = len(temp_df2.index)
        temp_df3 = temp_df2[(temp_df2['y']==1)]
        target_1_count = len(temp_df3)
        perc_1 = target_1_count / cluster_total
        perc_contribution_to_0 = target_1_count / n_in_1
        perc_contribution_to_1 = (cluster_total - target_1_count) / n_in_0
        list_of_perc_1.append(perc_1)
        list_of_total_in_cluster.append(cluster_total)
        list_of_perc_contribution_to_0.append(perc_contribution_to_0)
        list_of_perc_contribution_to_1.append(perc_contribution_to_1)
    array_of_clusters = np.array(list_of_clusters).astype('str')
    array_of_perc_1 = np.array(list_of_perc_1)
    array_of_contribution_to_1 = np.array(list_of_perc_contribution_to_1)
    array_of_contribution_to_0 = np.array(list_of_perc_contribution_to_0)
    array_of_total_in_cluster = np.array(list_of_total_in_cluster)
    chart_bars(array_of_clusters, array_of_perc_1, prefix+' % With Label == 1 by Cluster '+title, output_location\
    ,xlabel='Cluster', ylabel='% of Cluster With Label 1')
    chart_bars(array_of_clusters, array_of_total_in_cluster, prefix+' Total In Cluster '+title, output_location\
    ,xlabel='Cluster', ylabel='Rows')
    chart_bars(array_of_clusters, array_of_contribution_to_1, prefix+' % In Each Cluster For Label == 1 '+title, output_location\
    ,xlabel='Cluster', ylabel='% of Rows With Label 1')
    chart_bars(array_of_clusters, array_of_contribution_to_0, prefix+' % In Each Cluster For Label == 0 '+title, output_location\
    ,xlabel='Cluster', ylabel='% of Rows With Label 0')
    return temp_df


def run_dm_and_select_clusters(dataset_1, prefix, random_state, output_location, X_train, X_test, y_train, y_test):
    ## Built after running selection charts etc.
    if dataset_1 == True:
        pca = PCA(n_components=6, copy=True, whiten=False, svd_solver='auto', tol=0.0\
        , iterated_power='auto', random_state=random_state)
        ica = FastICA(n_components=None, algorithm='parallel', whiten=True, fun='logcosh', fun_args=None\
        , max_iter=200, tol=0.0001, w_init=None, random_state=random_state)
        rand_proj = GaussianRandomProjection(n_components=7,  eps=0.1, random_state=random_state)
        rfe_svr = RFE(estimator=SVR(kernel="linear"), n_features_to_select=6)
        rfe_dt = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=6)
        ica_kur_low_minimum = 1
        ica_kur_high_minimum = 3
    else:
        pca = PCA(n_components=8, copy=True, whiten=False, svd_solver='auto', tol=0.0\
        , iterated_power='auto', random_state=random_state)
        ica = FastICA(n_components=None, algorithm='parallel', whiten=True, fun='logcosh', fun_args=None\
        , max_iter=200, tol=0.0001, w_init=None, random_state=random_state)
        rand_proj = GaussianRandomProjection(n_components=9,  eps=0.1, random_state=random_state)
        rfe_svr = RFE(estimator=SVR(kernel="linear"), n_features_to_select=9)
        rfe_dt = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=9)
        ica_kur_low_minimum = 10
        ica_kur_high_minimum = 100
    #PCA
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    #ICA
    X_train_ica_full = ica.fit(X_train)
    X_train_ica_full = ica.transform(X_train)
    X_test_ica_full = ica.transform(X_test)
    kur = kurtosis(X_train_ica_full, fisher=True)
    print("kurtosis: ", kur)
    ica_mask_low = kur > ica_kur_low_minimum
    ica_mask_high = kur > ica_kur_high_minimum
    print("ica_masks: ")
    print(ica_mask_low,ica_mask_high)
    X_train_ica_high = X_train_ica_full[:,ica_mask_high]
    X_train_ica_low = X_train_ica_full[:,ica_mask_low]
    X_test_ica_high = X_test_ica_full[:,ica_mask_high]
    X_test_ica_low = X_test_ica_full[:,ica_mask_low]
    #RP
    rand_proj.fit(X_train)
    X_train_rand = rand_proj.transform(X_train)
    X_test_rand = rand_proj.transform(X_test)
    #RFE
    rfe_svr.fit(X_train, y_train)
    X_train_rfe_svr = rfe_svr.transform(X_train)
    X_test_rfe_svr = rfe_svr.transform(X_test)
    print("rfe.support_",rfe_svr.support_)
    print("rfe.ranking_",rfe_svr.ranking_)
    rfe_dt.fit(X_train, y_train)
    X_train_rfe_dt = rfe_dt.transform(X_train)
    X_test_rfe_dt = rfe_dt.transform(X_test)
    print("rfe.support_",rfe_dt.support_)
    print("rfe.ranking_",rfe_dt.ranking_)
    #Now create the artifacts
    pca_km_df = kmeans_by_k(X_train_pca, output_location+'pca/', prefix=prefix, title='PCA', n_init=10\
    , max_iter=300, tol=0.0001, verbose=0, random_state=random_state, copy_x=True,  algorithm='auto')
    pca_EM_df = EM_by_k(X_train_pca, output_location+'pca/', prefix=prefix, title='PCA', random_state=random_state)

    ica_h_km_df = kmeans_by_k(X_train_ica_high, output_location+'ica/', prefix=prefix, title='ICA', n_init=10\
    , max_iter=300, tol=0.0001, verbose=0, random_state=random_state, copy_x=True,  algorithm='auto')
    ica_h_EM_df = EM_by_k(X_train_ica_high, output_location+'ica/', prefix=prefix, title='ICA', random_state=random_state)
    ## steel_note

    ica_l_km_df = kmeans_by_k(X_train_ica_low, output_location+'ica/', prefix=prefix, title='ICA Low', n_init=10\
    , max_iter=300, tol=0.0001, verbose=0, random_state=random_state, copy_x=True,  algorithm='auto')
    ica_l_EM_df = EM_by_k(X_train_ica_low, output_location+'ica/', prefix=prefix, title='ICA Low', random_state=random_state)

    rp_km_df = kmeans_by_k(X_train_rand, output_location+'rp/', prefix=prefix, title='Rand Proj', n_init=10\
    , max_iter=300, tol=0.0001, verbose=0, random_state=random_state, copy_x=True,  algorithm='auto')
    rp_EM_df = EM_by_k(X_train_rand, output_location+'rp/', prefix=prefix, title='Rand Proj', random_state=random_state)

    rfe_dt_km_df = kmeans_by_k(X_train_rfe_dt, output_location+'rfe/', prefix=prefix, title='RFE', n_init=10\
    , max_iter=300, tol=0.0001, verbose=0, random_state=random_state, copy_x=True,  algorithm='auto')
    rfe_dt_EM_df = EM_by_k(X_train_rfe_dt, output_location+'rfe/', prefix=prefix, title='RFE', random_state=random_state)
    ## steel_note

    rfe_svr_km_df = kmeans_by_k(X_train_rfe_svr, output_location+'rfe/', prefix=prefix, title='RFE (SVR)', n_init=10\
    , max_iter=300, tol=0.0001, verbose=0, random_state=random_state, copy_x=True,  algorithm='auto')
    rfe_svr_EM_df = EM_by_k(X_train_rfe_svr, output_location+'rfe/', prefix=prefix, title='RFE (SVR)', random_state=random_state)

    # pca_df = analyze_clusters(kmeans, X_train_pca, y_train, output_location+'/pca', prefix=prefix, title="KMeans PCA")
    # ica_low_df = analyze_clusters(kmeans, X_train_ica_low, y_train, output_location+'/ica', prefix=prefix, title="KMeans ICA Low")
    # ica_high_df = analyze_clusters(kmeans, X_train_ica_high, y_train, output_location+'/ica', prefix=prefix, title="KMeans ICA")
    # rp_df = analyze_clusters(kmeans, X_train_rand, y_train, output_location+'/rp', prefix=prefix, title="KMeans Rand Proj")
    # rfe_svr_df = analyze_clusters(kmeans, X_train_rfe_dt, y_train, output_location+'/rfe', prefix=prefix, title="KMeans RFE")
    # rfe_dt_df = analyze_clusters(kmeans, X_train_rfe_svr, y_train, output_location+'/rfe', prefix=prefix, title="KMeans RFE (SVR)")
    return


def run_dm_cluster_analysis(dataset_1, prefix, random_state, output_location, X_train, X_test, y_train, y_test):
    ## Built after running selection charts etc.
    if dataset_1 == True:
        pca = PCA(n_components=6, copy=True, whiten=False, svd_solver='auto', tol=0.0\
        , iterated_power='auto', random_state=random_state)
        ica = FastICA(n_components=None, algorithm='parallel', whiten=True, fun='logcosh', fun_args=None\
        , max_iter=200, tol=0.0001, w_init=None, random_state=random_state)
        rand_proj = GaussianRandomProjection(n_components=7,  eps=0.1, random_state=random_state)
        rfe_svr = RFE(estimator=SVR(kernel="linear"), n_features_to_select=6)
        rfe_dt = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=6)
        ica_kur_low_minimum = 1
        ica_kur_high_minimum = 3
    else:
        pca = PCA(n_components=8, copy=True, whiten=False, svd_solver='auto', tol=0.0\
        , iterated_power='auto', random_state=random_state)
        ica = FastICA(n_components=None, algorithm='parallel', whiten=True, fun='logcosh', fun_args=None\
        , max_iter=200, tol=0.0001, w_init=None, random_state=random_state)
        rand_proj = GaussianRandomProjection(n_components=9,  eps=0.1, random_state=random_state)
        rfe_svr = RFE(estimator=SVR(kernel="linear"), n_features_to_select=9)
        rfe_dt = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=9)
        ica_kur_low_minimum = 10
        ica_kur_high_minimum = 100
    #PCA
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    #ICA
    X_train_ica_full = ica.fit(X_train)
    X_train_ica_full = ica.transform(X_train)
    X_test_ica_full = ica.transform(X_test)
    kur = kurtosis(X_train_ica_full, fisher=True)
    print("kurtosis: ", kur)
    ica_mask_low = kur > ica_kur_low_minimum
    ica_mask_high = kur > ica_kur_high_minimum
    print("ica_masks: ")
    print(ica_mask_low,ica_mask_high)
    X_train_ica_high = X_train_ica_full[:,ica_mask_high]
    X_train_ica_low = X_train_ica_full[:,ica_mask_low]
    X_test_ica_high = X_test_ica_full[:,ica_mask_high]
    X_test_ica_low = X_test_ica_full[:,ica_mask_low]
    #RP
    rand_proj.fit(X_train)
    X_train_rand = rand_proj.transform(X_train)
    X_test_rand = rand_proj.transform(X_test)
    #RFE
    rfe_svr.fit(X_train, y_train)
    X_train_rfe_svr = rfe_svr.transform(X_train)
    X_test_rfe_svr = rfe_svr.transform(X_test)
    print("rfe.support_",rfe_svr.support_)
    print("rfe.ranking_",rfe_svr.ranking_)
    rfe_dt.fit(X_train, y_train)
    X_train_rfe_dt = rfe_dt.transform(X_train)
    X_test_rfe_dt = rfe_dt.transform(X_test)
    print("rfe.support_",rfe_dt.support_)
    print("rfe.ranking_",rfe_dt.ranking_)
    #Now create the artifacts
    if dataset_1 == True:
        kmeans_pca = KMeans(n_clusters=9, verbose=0, random_state=random_state, copy_x=True)
        kmeans_ica_low = KMeans(n_clusters=10, verbose=0, random_state=random_state, copy_x=True)
        kmeans_ica_high = KMeans(n_clusters=8, verbose=0, random_state=random_state, copy_x=True)
        kmeans_rp = KMeans(n_clusters=10, verbose=0, random_state=random_state, copy_x=True)
        kmeans_rfe_svr = KMeans(n_clusters=9, verbose=0, random_state=random_state, copy_x=True)
        kmeans_rfe_dt = KMeans(n_clusters=7, verbose=0, random_state=random_state, copy_x=True)

        EMax_pca = GaussianMixture(n_components=7, random_state=random_state,  verbose=0)
        EMax_ica_low = GaussianMixture(n_components=3, random_state=random_state,  verbose=0)
        EMax_ica_high = GaussianMixture(n_components=6, random_state=random_state,  verbose=0)
        EMax_rp = GaussianMixture(n_components=5, random_state=random_state,  verbose=0)
        EMax_rfe_svr = GaussianMixture(n_components=4, random_state=random_state,  verbose=0)
        EMax_rfe_dt = GaussianMixture(n_components=5, random_state=random_state,  verbose=0)

    else:
        kmeans_pca = KMeans(n_clusters=3, verbose=0, random_state=random_state, copy_x=True)
        kmeans_ica_low = KMeans(n_clusters=12, verbose=0, random_state=random_state, copy_x=True)
        kmeans_ica_high = KMeans(n_clusters=8, verbose=0, random_state=random_state, copy_x=True)
        kmeans_rp = KMeans(n_clusters=7, verbose=0, random_state=random_state, copy_x=True)
        kmeans_rfe_svr = KMeans(n_clusters=5, verbose=0, random_state=random_state, copy_x=True)
        kmeans_rfe_dt = KMeans(n_clusters=3, verbose=0, random_state=random_state, copy_x=True)

        EMax_pca = GaussianMixture(n_components=9, random_state=random_state,  verbose=0)
        EMax_ica_low = GaussianMixture(n_components=7, random_state=random_state,  verbose=0)
        EMax_ica_high = GaussianMixture(n_components=10, random_state=random_state,  verbose=0)
        EMax_rp = GaussianMixture(n_components=8, random_state=random_state,  verbose=0)
        EMax_rfe_svr = GaussianMixture(n_components=7, random_state=random_state,  verbose=0)
        EMax_rfe_dt = GaussianMixture(n_components=7, random_state=random_state,  verbose=0)


    d = analyze_clusters(kmeans_pca, X_train_pca, y_train, output_location+'pca/', prefix=prefix, title="KMeans PCA")
    d = analyze_clusters(kmeans_ica_low, X_train_ica_low, y_train, output_location+'ica/', prefix=prefix, title="KMeans ICA Low")
    d = analyze_clusters(kmeans_ica_high, X_train_ica_high, y_train, output_location+'ica/', prefix=prefix, title="KMeans ICA")
    d = analyze_clusters(kmeans_rp, X_train_rand, y_train, output_location+'rp/', prefix=prefix, title="KMeans Rand Proj")
    d = analyze_clusters(kmeans_rfe_svr, X_train_rfe_dt, y_train, output_location+'rfe/', prefix=prefix, title="KMeans RFE")
    d = analyze_clusters(kmeans_rfe_dt, X_train_rfe_svr, y_train, output_location+'rfe/', prefix=prefix, title="KMeans RFE (SVR)")

    p = prefix
    d = analyze_clusters(EMax_pca, X_train_pca, y_train, output_location+'pca/', prefix=p, model="EM", title="KMeans PCA")
    d = analyze_clusters(EMax_ica_low, X_train_ica_low, y_train, output_location+'ica/', prefix=p, model="EM", title="KMeans ICA Low")
    d = analyze_clusters(EMax_ica_high, X_train_ica_high, y_train, output_location+'ica/', prefix=p, model="EM", title="KMeans ICA")
    d = analyze_clusters(EMax_rp, X_train_rand, y_train, output_location+'rp/', prefix=p, model="EM", title="KMeans Rand Proj")
    d = analyze_clusters(EMax_rfe_svr, X_train_rfe_dt, y_train, output_location+'rfe/', prefix=p, model="EM", title="KMeans RFE")
    d = analyze_clusters(EMax_rfe_dt, X_train_rfe_svr, y_train, output_location+'rfe/', prefix=p, model="EM", title="KMeans RFE (SVR)")
    return


def clustered_NN_grid_search(dataset_1, prefix, random_state, output_location, X_train, X_test, y_train, y_test):
    if dataset_1 == False:
        print("Only grid searching NN for dataset_1")
        return
    pca = PCA(n_components=6, copy=True, whiten=False, svd_solver='auto', tol=0.0\
    , iterated_power='auto', random_state=random_state)
    ica = FastICA(n_components=None, algorithm='parallel', whiten=True, fun='logcosh', fun_args=None\
    , max_iter=200, tol=0.0001, w_init=None, random_state=random_state)
    rand_proj = GaussianRandomProjection(n_components=7,  eps=0.1, random_state=random_state)
    rfe_svr = RFE(estimator=SVR(kernel="linear"), n_features_to_select=6)
    rfe_dt = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=6)
    ica_kur_low_minimum = 1
    ica_kur_high_minimum = 3

    #PCA
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    #ICA
    X_train_ica_full = ica.fit(X_train)
    X_train_ica_full = ica.transform(X_train)
    X_test_ica_full = ica.transform(X_test)
    kur = kurtosis(X_train_ica_full, fisher=True)
    print("kurtosis: ", kur)
    ica_mask_low = kur > ica_kur_low_minimum
    ica_mask_high = kur > ica_kur_high_minimum
    print("ica_masks: ")
    print(ica_mask_low,ica_mask_high)
    X_train_ica_high = X_train_ica_full[:,ica_mask_high]
    X_train_ica_low = X_train_ica_full[:,ica_mask_low]
    X_test_ica_high = X_test_ica_full[:,ica_mask_high]
    X_test_ica_low = X_test_ica_full[:,ica_mask_low]
    #RP
    rand_proj.fit(X_train)
    X_train_rand = rand_proj.transform(X_train)
    X_test_rand = rand_proj.transform(X_test)
    #RFE
    rfe_svr.fit(X_train, y_train)
    X_train_rfe_svr = rfe_svr.transform(X_train)
    X_test_rfe_svr = rfe_svr.transform(X_test)
    print("rfe.support_",rfe_svr.support_)
    print("rfe.ranking_",rfe_svr.ranking_)
    rfe_dt.fit(X_train, y_train)
    X_train_rfe_dt = rfe_dt.transform(X_train)
    X_test_rfe_dt = rfe_dt.transform(X_test)
    print("rfe.support_",rfe_dt.support_)
    print("rfe.ranking_",rfe_dt.ranking_)


    kmeans_pca = KMeans(n_clusters=9, verbose=0, random_state=random_state, copy_x=True)
    kmeans_ica_low = KMeans(n_clusters=10, verbose=0, random_state=random_state, copy_x=True)
    kmeans_ica_high = KMeans(n_clusters=8, verbose=0, random_state=random_state, copy_x=True)
    kmeans_rp = KMeans(n_clusters=10, verbose=0, random_state=random_state, copy_x=True)
    kmeans_rfe_svr = KMeans(n_clusters=9, verbose=0, random_state=random_state, copy_x=True)
    kmeans_rfe_dt = KMeans(n_clusters=7, verbose=0, random_state=random_state, copy_x=True)

    EMax_pca = GaussianMixture(n_components=7, random_state=random_state,  verbose=0)
    EMax_ica_low = GaussianMixture(n_components=3, random_state=random_state,  verbose=0)
    EMax_ica_high = GaussianMixture(n_components=6, random_state=random_state,  verbose=0)
    EMax_rp = GaussianMixture(n_components=5, random_state=random_state,  verbose=0)
    EMax_rfe_svr = GaussianMixture(n_components=4, random_state=random_state,  verbose=0)
    EMax_rfe_dt = GaussianMixture(n_components=5, random_state=random_state,  verbose=0)

    x_km_pca = kmeans_pca.fit_transform(X_train_pca)
    x_km_ica_low = kmeans_ica_low.fit_transform(X_train_ica_low)
    x_km_ica_high = kmeans_ica_high.fit_transform(X_train_ica_high)
    x_km_rp = kmeans_rp.fit_transform(X_train_rand)
    x_km_rfe_svr = kmeans_rfe_svr.fit_transform(X_train_rfe_svr)
    x_km_rfe_dt = kmeans_rfe_dt.fit_transform(X_train_rfe_dt)

    test_x_km_pca = kmeans_pca.transform(X_test_pca)
    test_x_km_ica_low = kmeans_ica_low.transform(X_test_ica_low)
    test_x_km_ica_high = kmeans_ica_high.transform(X_test_ica_high)
    test_x_km_rp = kmeans_rp.transform(X_test_rand)
    test_x_km_rfe_svr = kmeans_rfe_svr.transform(X_test_rfe_svr)
    test_x_km_rfe_dt = kmeans_rfe_dt.transform(X_test_rfe_dt)

    EMax_pca.fit(X_train_pca)
    EMax_ica_low.fit(X_train_ica_low)
    EMax_ica_high.fit(X_train_ica_high)
    EMax_rp.fit(X_train_rand)
    EMax_rfe_svr.fit(X_train_rfe_svr)
    EMax_rfe_dt.fit(X_train_rfe_dt)

    x_em_pca = EMax_pca.predict_proba(X_train_pca)
    x_em_ica_low = EMax_ica_low.predict_proba(X_train_ica_low)
    x_em_ica_high = EMax_ica_high.predict_proba(X_train_ica_high)
    x_em_rp = EMax_rp.predict_proba(X_train_rand)
    x_em_rfe_svr = EMax_rfe_svr.predict_proba(X_train_rfe_svr)
    x_em_rfe_dt = EMax_rfe_dt.predict_proba(X_train_rfe_dt)

    test_x_em_pca = EMax_pca.predict_proba(X_test_pca)
    test_x_em_ica_low = EMax_ica_low.predict_proba(X_test_ica_low)
    test_x_em_ica_high = EMax_ica_high.predict_proba(X_test_ica_high)
    test_x_em_rp = EMax_rp.predict_proba(X_test_rand)
    test_x_em_rfe_svr = EMax_rfe_svr.predict_proba(X_test_rfe_svr)
    test_x_em_rfe_dt = EMax_rfe_dt.predict_proba(X_test_rfe_dt)

    nn_param_grid = {
        'activation' : [ "relu",  "tanh", "identity", "logistic"],  #"identity", "logistic",
        'alpha' :  [0.001, 0.0001, 0.01],  #0.00001, 0.900005, 0.0001, 0.0005, 0.001, 0.1, 0.01, , 0.2
        'hidden_layer_sizes': [(5,), (5,2), (4,), (3,), (3,3), (10,), (8,), (12)], #[(10,), (5,),(6,),(4,)],#, (20,), (
        'max_iter' : [500], #, 700, 1000],  #100,200,
        'verbose' : [False],
        'random_state' : [random_state]
    }

    NN = neural_network.MLPClassifier()
    o = output_location
    # grid_search_print(nn_param_grid, 'NN_pca'+'_KM', NN, x_km_pca, y_train, o+'NN/', use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, 'NN_ica_low'+'_KM', NN, x_km_ica_low, y_train, o+'NN/', use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, 'NN_ica_high'+'_KM', NN, x_km_ica_high, y_train, o+'NN/', use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, 'NN_rp'+'_KM', NN, x_km_rp, y_train, o+'NN/', use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, 'NN_rfe_svr'+'_KM', NN, x_km_rfe_svr, y_train, o+'NN/', use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, 'NN_rfe_dt'+'_KM', NN, x_km_rfe_dt, y_train, o+'NN/', use_f1=False, prefix=prefix)


    # grid_search_print(nn_param_grid, 'NN_pca'+'_EM', NN, x_em_pca, y_train, o+'NN/', use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, 'NN_ica_low'+'_EM', NN, x_em_ica_low, y_train, o+'NN/', use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, 'NN_ica_high'+'_EM', NN, x_em_ica_high, y_train, o+'NN/', use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, 'NN_rp'+'_EM', NN, x_em_rp, y_train, o+'NN/', use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, 'NN_rfe_svr'+'_EM', NN, x_em_rfe_svr, y_train, o+'NN/', use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, 'NN_rfe_dt'+'_EM', NN, x_em_rfe_dt, y_train, o+'NN/', use_f1=False, prefix=prefix)

    nn_param_grid = {
        'activation' : [ "relu",  "tanh", "identity", "logistic"],  #"identity", "logistic",
        'alpha' :  [0.001, 0.0001, 0.01],  #0.00001, 0.900005, 0.0001, 0.0005, 0.001, 0.1, 0.01, , 0.2
        'hidden_layer_sizes': [(5,)], #(5,2), (4,), (3,), (3,3), (10,), (8,), (12)], #[(10,), (5,),(6,),(4,)],#, (20,), (
        'max_iter' : [500], #, 700, 1000],  #100,200,
        'verbose' : [False],
        'random_state' : [random_state]
    }

    # grid_search_print(nn_param_grid, 'SET5_NN_pca'+'_KM', NN, x_km_pca, y_train, o+'NN/', use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, 'SET5_NN_ica_low'+'_KM', NN, x_km_ica_low, y_train, o+'NN/', use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, 'SET5_NN_ica_high'+'_KM', NN, x_km_ica_high, y_train, o+'NN/', use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, 'SET5_NN_rp'+'_KM', NN, x_km_rp, y_train, o+'NN/', use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, 'SET5_NN_rfe_svr'+'_KM', NN, x_km_rfe_svr, y_train, o+'NN/', use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, 'SET5_NN_rfe_dt'+'_KM', NN, x_km_rfe_dt, y_train, o+'NN/', use_f1=False, prefix=prefix)


    # grid_search_print(nn_param_grid, 'SET5_NN_pca'+'_EM', NN, x_em_pca, y_train, o+'NN/', use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, 'SET5_NN_ica_low'+'_EM', NN, x_em_ica_low, y_train, o+'NN/', use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, 'SET5_NN_ica_high'+'_EM', NN, x_em_ica_high, y_train, o+'NN/', use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, 'SET5_NN_rp'+'_EM', NN, x_em_rp, y_train, o+'NN/', use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, 'SET5_NN_rfe_svr'+'_EM', NN, x_em_rfe_svr, y_train, o+'NN/', use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, 'SET5_NN_rfe_dt'+'_EM', NN, x_em_rfe_dt, y_train, o+'NN/', use_f1=False, prefix=prefix)

    NN_original_data = neural_network.MLPClassifier(activation='relu', alpha=0.001,
              hidden_layer_sizes=(5,), learning_rate='constant',
              random_state=random_state*2, verbose=False, warm_start=False)

    NN_km_pca = neural_network.MLPClassifier(activation='tanh', alpha=0.001,
              hidden_layer_sizes=(12,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    NN_km_ica_low = neural_network.MLPClassifier(activation='tanh', alpha=0.01,
              hidden_layer_sizes=(12,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    NN_km_ica_high = neural_network.MLPClassifier(activation='identity', alpha=0.01,
              hidden_layer_sizes=(12,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    NN_km_rand_proj = neural_network.MLPClassifier(activation='tanh', alpha=0.001,
              hidden_layer_sizes=(8,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    NN_km_rfe_dt = neural_network.MLPClassifier(activation='relu', alpha=0.001,
              hidden_layer_sizes=(4,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    NN_km_rfe_svr = neural_network.MLPClassifier(activation='tanh', alpha=0.01,
              hidden_layer_sizes=(12,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)

    NN_em_pca = neural_network.MLPClassifier(activation='identity', alpha=0.01,
              hidden_layer_sizes=(5,2), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    NN_em_ica_low = neural_network.MLPClassifier(activation='tanh', alpha=0.001,
              hidden_layer_sizes=(5,2), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    NN_em_ica_high = neural_network.MLPClassifier(activation='identity', alpha=0.001,
              hidden_layer_sizes=(3,3), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    NN_em_rand_proj = neural_network.MLPClassifier(activation='identity', alpha=0.001,
              hidden_layer_sizes=(3,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    NN_em_rfe_dt = neural_network.MLPClassifier(activation='identity', alpha=0.001,
              hidden_layer_sizes=(8,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    NN_em_rfe_svr = neural_network.MLPClassifier(activation='tanh', alpha=0.001,
              hidden_layer_sizes=(12,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)






    SET5_NN_km_pca = neural_network.MLPClassifier(activation='tanh', alpha=0.001,
              hidden_layer_sizes=(5,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    SET5_NN_km_ica_low = neural_network.MLPClassifier(activation='identity', alpha=0.01,
              hidden_layer_sizes=(5,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    SET5_NN_km_ica_high = neural_network.MLPClassifier(activation='relu', alpha=0.01,
              hidden_layer_sizes=(5,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    SET5_NN_km_rand_proj = neural_network.MLPClassifier(activation='relu', alpha=0.001,
              hidden_layer_sizes=(5,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    SET5_NN_km_rfe_dt = neural_network.MLPClassifier(activation='identity', alpha=0.001,
              hidden_layer_sizes=(5,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    SET5_NN_km_rfe_svr = neural_network.MLPClassifier(activation='logistic', alpha=0.01,
              hidden_layer_sizes=(5,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)

    SET5_NN_em_pca = neural_network.MLPClassifier(activation='tanh', alpha=0.01,
              hidden_layer_sizes=(5,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    SET5_NN_em_ica_low = neural_network.MLPClassifier(activation='identity', alpha=0.001,
              hidden_layer_sizes=(5,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    SET5_NN_em_ica_high = neural_network.MLPClassifier(activation='relu', alpha=0.001,
              hidden_layer_sizes=(5,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    SET5_NN_em_rand_proj = neural_network.MLPClassifier(activation='tanh', alpha=0.001,
              hidden_layer_sizes=(5,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    SET5_NN_em_rfe_dt = neural_network.MLPClassifier(activation='relu', alpha=0.001,
              hidden_layer_sizes=(5,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    SET5_NN_em_rfe_svr = neural_network.MLPClassifier(activation='logistic', alpha=0.001,
              hidden_layer_sizes=(5,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)

    nn_df_original = get_NN_results(NN_original_data, X_train, y_train, X_test, y_test, 'original')
    print(nn_df_original)

    nn_df_pca_km = get_NN_results(NN_km_pca, x_km_pca, y_train, test_x_km_pca, y_test, 'pca_km')
    nn_df_ica_low_km = get_NN_results(NN_km_ica_low, x_km_ica_low, y_train, test_x_km_ica_low, y_test, 'ica_low_km')
    nn_df_ica_high_km = get_NN_results(NN_km_ica_high, x_km_ica_high, y_train, test_x_km_ica_high, y_test, 'ica_high_km')
    nn_df_rp_km = get_NN_results(NN_km_rand_proj, x_km_rp, y_train, test_x_km_rp, y_test, 'rp_km')
    nn_df_rfe_dt_km = get_NN_results(NN_km_rfe_dt, x_km_rfe_dt, y_train, test_x_km_rfe_dt, y_test, 'rfe_dt_km')
    nn_df_rfe_svr_km = get_NN_results(NN_km_rfe_svr, x_km_rfe_svr, y_train, test_x_km_rfe_svr, y_test, 'rfe_svr_km')

    nn_df_pca_em = get_NN_results(NN_em_pca, x_em_pca, y_train, test_x_em_pca, y_test, 'pca_em')
    nn_df_ica_low_em = get_NN_results(NN_em_ica_low, x_em_ica_low, y_train, test_x_em_ica_low, y_test, 'ica_low_em')
    nn_df_ica_high_em = get_NN_results(NN_em_ica_high, x_em_ica_high, y_train, test_x_em_ica_high, y_test, 'ica_high_em')
    nn_df_rp_em = get_NN_results(NN_em_rand_proj, x_em_rp, y_train, test_x_em_rp, y_test, 'rp_em')
    nn_df_rfe_dt_em = get_NN_results(NN_em_rfe_dt, x_em_rfe_dt, y_train, test_x_em_rfe_dt, y_test, 'rfe_dt_em')
    nn_df_rfe_svr_em = get_NN_results(NN_em_rfe_svr, x_em_rfe_svr, y_train, test_x_em_rfe_svr, y_test, 'rfe_svr_em')

    nn_km_list = [nn_df_original, nn_df_pca_km, nn_df_ica_low_km, nn_df_ica_high_km, nn_df_rp_km, nn_df_rfe_dt_km, nn_df_rfe_svr_km]
    nn_em_list = [nn_df_original, nn_df_pca_em, nn_df_ica_low_em, nn_df_ica_high_em, nn_df_rp_em, nn_df_rfe_dt_em, nn_df_rfe_svr_em]
    nn_df_km = pd.concat(nn_km_list, ignore_index=True)
    nn_df_em = pd.concat(nn_em_list, ignore_index=True)
    print("\n\n\n\n\n\n\n",nn_df_km)
    print("\n\n\n\n\n\n\n",nn_df_em)

    nn_df_km.to_csv(output_location+'NN/'+'nn_df_km.csv')
    nn_df_em.to_csv(output_location+'NN/'+'nn_df_em.csv')

    nn_df_pca_km = get_NN_results(SET5_NN_km_pca, x_km_pca, y_train, test_x_km_pca, y_test, 'pca_km')
    nn_df_ica_low_km = get_NN_results(SET5_NN_km_ica_low, x_km_ica_low, y_train, test_x_km_ica_low, y_test, 'ica_low_km')
    nn_df_ica_high_km = get_NN_results(SET5_NN_km_ica_high, x_km_ica_high, y_train, test_x_km_ica_high, y_test, 'ica_high_km')
    nn_df_rp_km = get_NN_results(SET5_NN_km_rand_proj, x_km_rp, y_train, test_x_km_rp, y_test, 'rp_km')
    nn_df_rfe_dt_km = get_NN_results(SET5_NN_km_rfe_dt, x_km_rfe_dt, y_train, test_x_km_rfe_dt, y_test, 'rfe_dt_km')
    nn_df_rfe_svr_km = get_NN_results(SET5_NN_km_rfe_svr, x_km_rfe_svr, y_train, test_x_km_rfe_svr, y_test, 'rfe_svr_km')

    nn_df_pca_em = get_NN_results(SET5_NN_em_pca, x_em_pca, y_train, test_x_em_pca, y_test, 'pca_em')
    nn_df_ica_low_em = get_NN_results(SET5_NN_em_ica_low, x_em_ica_low, y_train, test_x_em_ica_low, y_test, 'ica_low_em')
    nn_df_ica_high_em = get_NN_results(SET5_NN_em_ica_high, x_em_ica_high, y_train, test_x_em_ica_high, y_test, 'ica_high_em')
    nn_df_rp_em = get_NN_results(SET5_NN_em_rand_proj, x_em_rp, y_train, test_x_em_rp, y_test, 'rp_em')
    nn_df_rfe_dt_em = get_NN_results(SET5_NN_em_rfe_dt, x_em_rfe_dt, y_train, test_x_em_rfe_dt, y_test, 'rfe_dt_em')
    nn_df_rfe_svr_em = get_NN_results(SET5_NN_em_rfe_svr, x_em_rfe_svr, y_train, test_x_em_rfe_svr, y_test, 'rfe_svr_em')

    nn_km_list = [nn_df_original, nn_df_pca_km, nn_df_ica_low_km, nn_df_ica_high_km, nn_df_rp_km, nn_df_rfe_dt_km, nn_df_rfe_svr_km]
    nn_em_list = [nn_df_original, nn_df_pca_em, nn_df_ica_low_em, nn_df_ica_high_em, nn_df_rp_em, nn_df_rfe_dt_em, nn_df_rfe_svr_em]
    nn_df_km_SET5 = pd.concat(nn_km_list, ignore_index=True)
    nn_df_em_SET5 = pd.concat(nn_em_list, ignore_index=True)
    print("\n\n\n\n\n\n\n",nn_df_km)
    print("\n\n\n\n\n\n\n",nn_df_em)

    nn_df_km_SET5.to_csv(output_location+'NN/'+'SET5_nn_df_km.csv')
    nn_df_em_SET5.to_csv(output_location+'NN/'+'SET5_nn_df_em.csv')




    Pima_NN_original_data = neural_network.MLPClassifier(activation='relu', alpha=0.001,
              hidden_layer_sizes=(5,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    Pima_NN_pca = neural_network.MLPClassifier(activation='tanh', alpha=0.01,
              hidden_layer_sizes=(10,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    Pima_NN_ica_low = neural_network.MLPClassifier(activation='tanh', alpha=0.001,
              hidden_layer_sizes=(10,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    Pima_NN_ica_high = neural_network.MLPClassifier(activation='relu', alpha=0.001,
              hidden_layer_sizes=(5,2), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    Pima_NN_rand_proj = neural_network.MLPClassifier(activation='identity', alpha=0.001,
              hidden_layer_sizes=(5,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    Pima_NN_rfe_dt = neural_network.MLPClassifier(activation='tanh', alpha=0.001,
              hidden_layer_sizes=(3,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)
    Pima_NN_rfe_svr = neural_network.MLPClassifier(activation='logistic', alpha=0.001,
              hidden_layer_sizes=(10,), learning_rate='constant',
              random_state=random_state, verbose=False, warm_start=False)

    nn_df_pca_km = get_NN_results(Pima_NN_pca, X_train_pca, y_train, X_test_pca, y_test, 'pca_km')
    nn_df_ica_low_km = get_NN_results(Pima_NN_ica_low, X_train_ica_low, y_train, X_test_ica_low, y_test, 'ica_low_km')
    nn_df_ica_high_km = get_NN_results(Pima_NN_ica_high, X_train_ica_high, y_train, X_test_ica_high, y_test, 'ica_high_km')
    nn_df_rp_km = get_NN_results(Pima_NN_rand_proj, X_train_rand, y_train, X_test_rand, y_test, 'rp_km')
    nn_df_rfe_dt_km = get_NN_results(Pima_NN_rfe_dt, X_train_rfe_dt, y_train, X_test_rfe_dt, y_test, 'rfe_dt_km')
    nn_df_rfe_svr_km = get_NN_results(Pima_NN_rfe_svr, X_train_rfe_svr, y_train, X_test_rfe_svr, y_test, 'rfe_svr_km')

    nn_km_list = [nn_df_original, nn_df_pca_km, nn_df_ica_low_km, nn_df_ica_high_km, nn_df_rp_km, nn_df_rfe_dt_km, nn_df_rfe_svr_km]
    nn_df_with_dr = pd.concat(nn_km_list, ignore_index=True)
    nn_df_with_dr.to_csv(output_location+'NN/'+'nn_df_with_dr.csv')

    names_list = list(nn_df_km['name'])
    names_array = np.array(names_list)
    accuracy_km = np.array(nn_df_km['test_accuracy'])
    accuracy_em = np.array(nn_df_em['test_accuracy'])
    accuracy_dr = np.array(nn_df_with_dr['test_accuracy'])
    time_km = np.array(nn_df_km['time'])
    time_em = np.array(nn_df_em['time'])
    time_dr = np.array(nn_df_with_dr['time'])

    chart_hbars(names_array, accuracy_km, prefix+" Dim Reduction KMeans NN Accuracy", output_location+"NN/"
    ,ylabel="NN Inputs", xlabel="Accuracy")
    chart_hbars(names_array, accuracy_em, prefix+" Dim Reduction EM NN Accuracy", output_location+"NN/"
    ,ylabel="NN Inputs", xlabel="Accuracy")
    chart_hbars(names_array, accuracy_dr, prefix+" Dim Reduction (No Cluster) NN Accuracy", output_location+"NN/"
    ,ylabel="NN Inputs", xlabel="Accuracy")
    chart_hbars(names_array, time_km, prefix+" Dim Reduction KMeans NN Fit Time", output_location+"NN/"
    ,ylabel="NN Inputs", xlabel="Time")
    chart_hbars(names_array, time_em, prefix+" Dim Reduction EM NN Fit Time", output_location+"NN/"
    ,ylabel="NN Inputs", xlabel="Time")
    chart_hbars(names_array, time_dr, prefix+" Dim Reduction (No Cluster) NN Fit Time", output_location+"NN/"
    ,ylabel="NN Inputs", xlabel="Time")


    names_list = list(nn_df_km['name'])
    names_array = np.array(names_list)
    accuracy_km = np.array(nn_df_km['test_accuracy'])
    accuracy_em = np.array(nn_df_em['test_accuracy'])
    accuracy_dr = np.array(nn_df_with_dr['test_accuracy'])
    time_km = np.array(nn_df_km['time'])
    time_em = np.array(nn_df_em['time'])
    time_dr = np.array(nn_df_with_dr['time'])

    chart_hbars(names_array, accuracy_km, prefix+" Dim Reduction KMeans NN Accuracy", output_location+"NN/"
    ,ylabel="NN Inputs", xlabel="Accuracy")
    chart_hbars(names_array, accuracy_em, prefix+" Dim Reduction EM NN Accuracy", output_location+"NN/"
    ,ylabel="NN Inputs", xlabel="Accuracy")
    chart_hbars(names_array, accuracy_dr, prefix+" Dim Reduction (No Cluster) NN Accuracy", output_location+"NN/"
    ,ylabel="NN Inputs", xlabel="Accuracy")
    chart_hbars(names_array, time_km, prefix+" Dim Reduction KMeans NN Fit Time", output_location+"NN/"
    ,ylabel="NN Inputs", xlabel="Time")
    chart_hbars(names_array, time_em, prefix+" Dim Reduction EM NN Fit Time", output_location+"NN/"
    ,ylabel="NN Inputs", xlabel="Time")
    chart_hbars(names_array, time_dr, prefix+" Dim Reduction (No Cluster) NN Fit Time", output_location+"NN/"
    ,ylabel="NN Inputs", xlabel="Time")

    names_array = np.array(names_list)
    accuracy_km = np.array(nn_df_km_SET5['test_accuracy'])
    accuracy_em = np.array(nn_df_em_SET5['test_accuracy'])
    accuracy_dr = np.array(nn_df_with_dr['test_accuracy'])
    time_km = np.array(nn_df_km_SET5['time'])
    time_em = np.array(nn_df_em_SET5['time'])
    time_dr = np.array(nn_df_with_dr['time'])

    chart_hbars(names_array, accuracy_km, prefix+" Dim Red KMeans NN Accuracy (No NN Structure Change)", output_location+"NN/"
    ,ylabel="NN Inputs", xlabel="Accuracy")
    chart_hbars(names_array, accuracy_em, prefix+" Dim Red EM NN Accuracy (No NN Structure Change)", output_location+"NN/"
    ,ylabel="NN Inputs", xlabel="Accuracy")
    chart_hbars(names_array, time_km, prefix+" Dim Reduction KMeans NN Fit Time (No NN Structure Change)", output_location+"NN/"
    ,ylabel="NN Inputs", xlabel="Time")
    chart_hbars(names_array, time_em, prefix+" Dim Reduction EM NN Fit Time (No NN Structure Change)", output_location+"NN/"
    ,ylabel="NN Inputs", xlabel="Time")

    return


def get_NN_results(NN, X_train, y_train, X_test, y_test, name):
    start_time_fit = time.perf_counter()
    NN.fit(X_train, y_train)
    end_time_fit = time.perf_counter()
    time_used = end_time_fit - start_time_fit
    #accuracy for train
    train_predictions = NN.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_f1 = f1_score(y_train, train_predictions)
    #accuracy for test
    test_predictions = NN.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions)
    temp_df4 = pd.DataFrame({'name':[name], 'time':[time_used], 'train_accuracy':[train_accuracy], 'train_f1':[train_f1]\
    ,'test_accuracy':[test_accuracy 'test_f1':[test_f1]})
    return temp_df4


def analyze_original_data_clusters(dataset_1, data, y_data, output_location, random_state=12345, prefix="", title=""):
    if dataset_1 == True:
        kmeans = KMeans(n_clusters=7, random_state=random_state)
        em = GaussianMixture(n_components=6, random_state=random_state)
    else:
        kmeans = KMeans(n_clusters=3, random_state=random_state)
        em = GaussianMixture(n_components=4, random_state=random_state)
    analyze_clusters(kmeans, data, y_data, output_location, model='kmeans', prefix=prefix, title="KM")
    analyze_clusters(em, data, y_data, output_location, model='em', prefix=prefix, title="EM")
    print("clusters on original data done been analyzed (and saved at the specified output_location)")
    if dataset_1 == True:
        kmeans = KMeans(n_clusters=9, random_state=random_state)
        em = GaussianMixture(n_components=6, random_state=random_state)
    else:
        kmeans = KMeans(n_clusters=4, random_state=random_state)
        em = GaussianMixture(n_components=4, random_state=random_state)
    analyze_clusters(kmeans, data, y_data, output_location, model='kmeans', prefix=prefix+' 2', title=title)
    print("clusters on original data done been analyzed (and saved at the specified output_location)")
    return


def chart_curve(x_array, y_array, title, output_location, labels_list=0\
, use_f1=False, xlabel=None, ylabel=None):
    plt.figure(figsize=(6.4*.5,4.8*.5))
    if y_array.ndim == 1:
        num_lines = 1
    else:
        num_lines = y_array.shape[0]
    if num_lines == 1:
        plt.plot(x_array, y_array, marker='o')
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
    plt.figure(figsize=(6.4*.5,4.8*.5))
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


def make_pair_plot():
    #sns.pairplot(df)
    #sns.pairplot(df, hue = 'continent')
    return


def main():
    dataset_1 =  False #True # (For convenience; I look at 2 data sets in this assignment)
    create_cluster_and_dimension_selection_charts = False #True #
    final_analysis = False #True #
    cluster_analysis_charts = True #False #
    print("Welcome to unsupervized learning. This will be great.\nInitializing inputs...")
    project_folder = '/unsupervised_learning/'
    input_location = project_folder + 'data/'
    pickles_location = project_folder + 'pickles/'
    random_state = 12345
    if dataset_1 == True:
        output_location = project_folder + 'outputs2/'
        chart_output_location = project_folder + 'charts/'
        use_f1 =  False #True #
        file_name = 'pima_diabetes.csv'
        target = "Outcome"
        prefix="PIMA"
    else:
        output_location = project_folder + 'outputs_amz2/'
        chart_output_location = project_folder + 'charts_amz/'
        use_f1 =  True #
        file_name = 'amazon_data3.csv'
        target = "low_score_ind"
        prefix="AMZN"
    df = pd.read_csv(input_location + file_name)
    X_train, X_test, y_train, y_test = split_and_scale(df, target, train_size=0.75, random_state=random_state)
    print("To begin, I split and scale the data. X_train has the shape:",np.shape(X_train))
    pca_vs_rand_reconstruction(X_train, output_location, random_state=random_state, title="",prefix=prefix)
    print("I am going to create all of the K selection charts for dim reductions")
    if final_analysis == True:
        pca_vs_rand_reconstruction(X_train, output_location, random_state=random_state, title="",prefix=prefix)
        run_dm_and_select_clusters(dataset_1, prefix, random_state, output_location, X_train, X_test, y_train, y_test)
        run_dm_cluster_analysis(dataset_1, prefix, random_state, output_location, X_train, X_test, y_train, y_test)
        clustered_NN_grid_search(dataset_1, prefix, random_state, output_location, X_train, X_test, y_train, y_test)

    if create_cluster_and_dimension_selection_charts == True:
        ## Using PCA
        pca = PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0\
        , iterated_power='auto', random_state=random_state)
        start_time_fit = time.perf_counter()
        pca.fit(X_train)
        end_time_fit = time.perf_counter()
        time_used_to_fit = end_time_fit - start_time_fit
        print("PCA Eigen Values...\n\n\n")
        print("That time was :",time_used_to_fit)
        create_PCA_charts(pca, output_location, prefix=prefix)

        ## Using ICA
        ica = FastICA(n_components=None, algorithm='parallel', whiten=True, fun='logcosh', fun_args=None\
        , max_iter=200, tol=0.0001, w_init=None, random_state=random_state)
        X_train_ica_full = ica.fit_transform(X_train)
        kur = kurtosis(X_train_ica_full, fisher=True)
        print("kurtosis: ", kur)
        ica_mask = kur > 1
        print("ica_mask ",ica_mask)
        features_list = list(range(1,len(kur)+1,1))
        print("charting the bars for kurtosis ... \n\n\n\n")
        chart_bars(list(range(1,len(kur)+1,1)), kur, prefix+' ICA Kurtosis by Feature', output_location\
        ,xlabel='Kurtosis', ylabel='Feature')

        ## Using Random Projection
        rand_proj_evaluation(X_train, output_location, prefix=prefix, random_state = random_state, title='' )

        #Can make a table to talk through differences in the RFE with different estimators
        rfe = RFE(estimator=SVR(kernel="linear"), n_features_to_select=3)
        rfe.fit(X_train, y_train)
        print("rfe.support_",rfe.support_)
        print("rfe.ranking_",rfe.ranking_)
        new_X = X_train[:,rfe.support_]

        rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=3)
        rfe.fit(X_train, y_train)
        print("rfe.support_",rfe.support_)
        print("rfe.ranking_",rfe.ranking_)

    #####################  Dim Reduct Inputs  ##################
    if cluster_analysis_charts == True:
        ## This was just for ease of reproducing results. This function has inputs obtained from analysis.
        analyze_original_data_clusters(dataset_1, X_train, y_train, output_location, random_state=random_state, prefix=prefix+" ", title="")


    ## Grid search parameters iteratively
    nn_param_grid = {
        'activation' : [ "relu",  "tanh", "identity", "logistic"],  #"identity", "logistic",
        'alpha' :  [0.001, 0.0001, 0.01],  #0.00001, 0.900005, 0.0001, 0.0005, 0.001, 0.1, 0.01, , 0.2
        'hidden_layer_sizes': [(5,), (5,2), (4,), (3,), (10,)], #[(10,), (5,),(6,),(4,)],#, (20,), (
        'max_iter' : [500], #, 700, 1000],  #100,200,
        'verbose' : [False],
        'random_state' : [random_state]
    }
    nn_param_grid_2 = {
        'activation' : [ "relu",  "tanh", "identity", "logistic"],  #"identity", "logistic",
        'alpha' :  [0.001],  #0.00001, 0.900005, 0.0001, 0.0005, 0.001, 0.1, 0.01, , 0.2
        'hidden_layer_sizes': [(5,), (10,), (15,)], #[(10,), (5,),(6,),(4,)],#, (20,), (100,10)], #,(12,),(10,5), (15,), (15,5)],
        'max_iter' : [500], #, 700, 1000],  #100,200,
        'verbose' : [False],
        'random_state' : [random_state]
    }

    # grid_search_print(nn_param_grid, 'NN_original_data', NN, X_train, y_train, output_location, use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, '2nd_NN_ica_2', NN, X_train_ica_2, y_train, output_location, use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, '2nd_NN_pca', NN, X_train_PCA, y_train, output_location, use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, '2nd_NN_ica_1', NN, X_train_ica_1, y_train, output_location, use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, '2nd_NN_ica_3', NN, X_train_ica_3, y_train, output_location, use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, '2nd_NN_rand_proj', NN, X_train_rand, y_train, output_location, use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, '2nd_NN_rfe_dt', NN, X_train_rfe_dt, y_train, output_location, use_f1=False, prefix=prefix)
    # grid_search_print(nn_param_grid, '2nd_NN_rfe_svr', NN, X_train_rfe_svr, y_train, output_location, use_f1=False, prefix=prefix)

    print("This is the end of the program. You are welcome.")
if __name__ == "__main__":
    main()
