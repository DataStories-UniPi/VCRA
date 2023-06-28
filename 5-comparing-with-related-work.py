import numpy as np
import pandas as pd
import geopandas as gpd

import argparse
import sys, os
import sklearn
import datetime

import importlib
from tqdm import tqdm
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns

import cri_calc as cri
importlib.reload(cri)

import cri_helper as helper
importlib.reload(helper)

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn_rvm import EMRVR

from sklearn.neural_network import MLPRegressor
pd.set_option('display.max_columns', None)


# %%
def evaluate_clf(clf, X, y, train_index, test_index, include_indices=False):
    print(f'Training with {len(train_index)} samples; Testing with {len(test_index)} samples')
    
    # Get Train/Test Sets
    X_train, X_test = X.iloc[train_index].values, X.iloc[test_index].values
    y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values

    # Train Model on Selected Fold
    clf.fit(X_train, y_train)
    y_pred = np.clip(clf.predict(X_test), 0, 1)
    
    # Organize and Return Results
    result = dict(
        instance = clf,
        train_indices=train_index,
        test_indices=test_index,
        y_true = y_test,
        y_pred = y_pred,
        acc = clf.score(X_test, y_test),
        mae = mean_absolute_error(y_test, y_pred),
        rmse = mean_squared_error(y_test, y_pred, squared=False),
        rmsle = mean_squared_log_error(y_test, y_pred, squared=False),
    )
    
    if include_indices:
        result.update({
            'train_indices':train_index,
            'test_indices':test_index
        })
        
    return result


# %% SVM-VCRA (Gang et al.)
def train_svm_vcra(X_sub, y_sub, y_bin_sub, tag=''):
    # svm_vcra_features = ['dist_euclid', 'own_speed', 'target_speed', 'own_course', 'target_course', 'relative_bearing_target_to_own']
    svm_vcra_features = ['dist_euclid', 'own_speed', 'target_speed', 'own_course_rad', 'target_course_rad', 'relative_bearing_target_to_own']
    svm_vcra_training_data = X_sub.loc[:, svm_vcra_features].copy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
    clf = make_pipeline(StandardScaler(), SVR(gamma='auto', kernel='rbf', verbose=True))
        
    svm_vcra_skf_results = Parallel(n_jobs=-1)(delayed(evaluate_clf)(
        clf, svm_vcra_training_data, y_sub, train_index, test_index
    ) for (train_index, test_index) in tqdm(skf.split(X_sub, y_bin_sub), total=skf.get_n_splits(X_sub, y_bin_sub)))

    svm_vcra_skf_results_df = pd.DataFrame(svm_vcra_skf_results)
    svm_vcra_skf_results_df.to_pickle(f'./data/pickle/svm_vcra_skf_results_v14{tag}.pickle')


# %% RVM-VCRA (Park et al.)
def train_rvm_vcra(X_sub, y_sub, y_bin_sub, tag=''):
    # rvm_vcra_features = ['dist_euclid', 'own_speed', 'target_speed', 'own_course', 'target_course', 'relative_bearing_target_to_own', 'own_length', 'target_length']
    rvm_vcra_features = ['dist_euclid', 'own_speed', 'target_speed', 'own_course_rad', 'target_course_rad', 'relative_bearing_target_to_own', 'own_length_nmi', 'target_length_nmi']
    rvm_vcra_training_data = X_sub.loc[:, rvm_vcra_features].copy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
    clf = make_pipeline(StandardScaler(), EMRVR(kernel='rbf', verbose=True))
        
    rvm_vcra_skf_results = Parallel(n_jobs=-1)(delayed(evaluate_clf)(
        clf, rvm_vcra_training_data, y_sub, test_index, train_index
    ) for (train_index, test_index) in tqdm(skf.split(X_sub, y_bin_sub), total=skf.get_n_splits(X_sub, y_bin_sub)))

    rvm_vcra_skf_results_df = pd.DataFrame(rvm_vcra_skf_results)
    rvm_vcra_skf_results_df.to_pickle(f'./data/pickle/rvm_vcra_skf_results_v14{tag}.pickle')


# %% CART-VCRA (Li et al.)
def train_cart_vcra(X_sub, y_sub, y_bin_sub, tag=''):
    # cart_vcra_features = ['dist_euclid', 'own_speed', 'target_speed', 'own_course', 'target_course', 'azimuth_angle_target_to_own']
    cart_vcra_features = ['dist_euclid', 'own_speed', 'target_speed', 'own_course_rad', 'target_course_rad', 'azimuth_angle_target_to_own']
    cart_vcra_training_data = X_sub.loc[:, cart_vcra_features].copy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
    clf = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=10, verbose=True))
        
    cart_vcra_skf_results = Parallel(n_jobs=-1)(delayed(evaluate_clf)(
        clf, cart_vcra_training_data, y_sub, train_index, test_index
    ) for (train_index, test_index) in tqdm(skf.split(X_sub, y_bin_sub), total=skf.get_n_splits(X_sub, y_bin_sub)))

    cart_vcra_skf_results_df = pd.DataFrame(cart_vcra_skf_results)
    cart_vcra_skf_results_df.to_pickle(f'./data/pickle/cart_vcra_skf_results_v14{tag}.pickle')


# %% MLP-VCRA (Ours)
def train_mlp_vcra(X_sub, y_sub, y_bin_sub, tag=''):
    # %% mlp_vcra_features = ['own_speed', 'own_course', 'target_speed', 'target_course', 'dist_euclid', 'azimuth_angle_target_to_own', 'rel_movement_direction']
    mlp_vcra_features = ['own_speed', 'own_course_rad', 'target_speed', 'target_course_rad', 'dist_euclid', 'azimuth_angle_target_to_own', 'rel_movement_direction']
    mlp_vcra_training_data = X_sub.loc[:, mlp_vcra_features].copy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
    regr = make_pipeline(
        StandardScaler(), 
        MLPRegressor(random_state=10, max_iter=300, hidden_layer_sizes=(256, 32), 
                    verbose=True, early_stopping=True, n_iter_no_change=7)
    )

    mlp_vcra_skf_results = Parallel(n_jobs=-1)(delayed(evaluate_clf)(
        regr, mlp_vcra_training_data, y_sub, train_index, test_index
    ) for (train_index, test_index) in tqdm(skf.split(X_sub, y_bin_sub), total=skf.get_n_splits(X_sub, y_bin_sub)))

    mlp_vcra_skf_results_df = pd.DataFrame(mlp_vcra_skf_results)
    mlp_vcra_skf_results_df.to_pickle(f'./data/pickle/mlp_vcra_skf_results_v14{tag}.pickle')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Train VCRA Model')
    parser.add_argument('--model', help='Select Model', default='mlp', choices=['svm', 'rvm', 'cart', 'mlp'])
    parser.add_argument('--use_subset', help='Use a stratified subset (for "RAM" hungry models)', action='store_true')
    args = parser.parse_args()


    # %% Loading and Preparing CRI Dataset
    gdf_vcra = pd.read_pickle('./data/norway-dataset/oslo_jan_mar_2019_4w_prep_encountering.vcra_dataset_v14.pickle')
    gdf_vcra.loc[:, 'ves_cri_bin'] = pd.cut(
        gdf_vcra.ves_cri, bins=np.arange(0, 1.1, .2),
        right=True, include_lowest=True
    )

    ves_cri_bin_val_counts = gdf_vcra.ves_cri_bin.value_counts(sort=False)
    print(ves_cri_bin_val_counts)
    ax = ves_cri_bin_val_counts.plot.bar()
    ax.set_yscale('log')
    plt.savefig('oslo_jan_mar_2019_4w_prep_encountering.ves_cri.distribution.pdf', dpi=300)

    # %% Get a Stratified Subset (to ensure a "fair" comparison)
    X, y, y_bin = gdf_vcra.iloc[:, :-2], gdf_vcra.iloc[:, -2], gdf_vcra.iloc[:, -1].astype('str')
    X_sub, _, y_sub, _, y_bin_sub, _ = train_test_split(X, y, y_bin, train_size=0.35, random_state=10, stratify=y_bin)

    fun_train = eval(f'train_{args.model}_vcra')
    args = (X_sub, y_sub, y_bin_sub) if args.use_subset else (X, y, y_bin, ".trained_on_all_data")
    fun_train(*args)
