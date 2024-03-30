#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 03:02:05 2024

@author: nado
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.externals import joblib
from joblib import Parallel, delayed

# Global Constants
NUM_TREES = 300

# Data Paths
TRAIN_X_PATH = ".../data/signatures_train.csv"
TRAIN_Y_PATH = ".../data/gene_expression_train.csv"
TEST_X_PATH = ".../data/signatures_test.csv"
TEST_Y_PATH = ".../data/gene_expression_test.csv"
SIM_RESULT_PATH = "Sim_result.csv"
MODEL_OUTPUT_PATH = ".../output/ML/"
META_OUTPUT_PATH = ".../output/ML/meta/"

# Read Data
train_X = pd.read_csv(TRAIN_X_PATH, index_col=0)
train_Y = pd.read_csv(TRAIN_Y_PATH, index_col=0)
test_X = pd.read_csv(TEST_X_PATH, index_col=0)
test_Y = pd.read_csv(TEST_Y_PATH, index_col=0)
similarity_results = pd.read_csv(SIM_RESULT_PATH, index_col=0)

# Extracting genes with activity
genes = similarity_results.index.tolist()

def predict_model_result(gene):
    regressor = joblib.load(f"{MODEL_OUTPUT_PATH}/FR/model/rfrg{gene}.pkl")
    train_predictions = regressor.predict(train_X)
    test_predictions = regressor.predict(test_X)
    return {'gene': f"{gene}_2", 'train_predictions': train_predictions, 'test_predictions': test_predictions}

def extract_extra_features(genes_to_include):
    features_train = pd.DataFrame(index=train_X.index)
    features_test = pd.DataFrame(index=test_X.index)
    result = Parallel(n_jobs=-1, verbose=1, pre_dispatch='all')(delayed(predict_model_result)(gene) for gene in genes_to_include)
    for res in result:
        features_train[res['gene']] = res['train_predictions']
        features_test[res['gene']] = res['test_predictions']
    return features_train, features_test

def train_first_round_models():
    for gene in genes:
        # Prepare Data
        X_temp = train_X.copy()
        X_temp['gene'] = gene
        X = pd.merge(X_temp, similarity_results, how='left', left_on=['gene'], right_index=True).drop(['gene'], axis=1)
        y = train_Y[gene]
        # Train Model
        regressor = RandomForestRegressor(n_estimators=NUM_TREES, n_jobs=-1)
        regressor.fit(X, np.ravel(y))
        # Make Predictions
        X_test_temp = test_X.copy()
        X_test_temp['gene'] = gene
        X_test = pd.merge(X_test_temp, similarity_results, how='left', left_on=['gene'], right_index=True).drop(['gene'], axis=1)
        predictions = regressor.predict(X_test)
        # Evaluate Model
        r2 = r2_score(test_Y[gene], predictions)
        # Save Results
        pd.DataFrame({'r2': [r2]}, index=[gene]).to_csv(f"{MODEL_OUTPUT_PATH}/FR/result_{gene}.csv")
        joblib.dump(regressor, f"{MODEL_OUTPUT_PATH}/FR/model/rfrg{gene}.pkl")
        pd.DataFrame(predictions).to_csv(f"{MODEL_OUTPUT_PATH}/FR/predictions/pred{gene}.csv")

def extract_metadata():
    for gene in genes:
        X_temp = train_X.copy()
        X_temp['gene'] = gene
        X = pd.merge(X_temp, similarity_results, how='left', left_on=['gene'], right_index=True).drop(['gene'], axis=1)
        y = train_Y[gene]
        X_test_temp = test_X.copy()
        X_test_temp['gene'] = gene
        X_test = pd.merge(X_test_temp, similarity_results, how='left', left_on=['gene'], right_index=True).drop(['gene'], axis=1)
        genes_to_include = genes[:]
        genes_to_include.remove(gene)
        features_train, features_test = extract_extra_features(genes_to_include)
        features_train.to_csv(f"{META_OUTPUT_PATH}/extra_feat_train_{gene}.csv")
        features_test.to_csv(f"{META_OUTPUT_PATH}/extra_feat_test_{gene}.csv")

def train_second_round_models():
    for gene in genes:
        y = train_Y[gene]
        features_train = pd.read_csv(f"{META_OUTPUT_PATH}/extra_feat_train_{gene}.csv", index_col=0)
        features_test = pd.read_csv(f"{META_OUTPUT_PATH}/extra_feat_test_{gene}.csv", index_col=0)
        X_temp = train_X.copy()
        X_temp['gene'] = gene
        X = pd.merge(X_temp, similarity_results, how='left', left_on=['gene'], right_index=True).drop(['gene'], axis=1)
        X_test_temp = test_X.copy()
        X_test_temp['gene'] = gene
        X_test = pd.merge(X_test_temp, similarity_results, how='left', left_on=['gene'], right_index=True).drop(['gene'], axis=1)
        train_data = pd.concat([features_train, X], axis=1)
        test_data = pd.concat([features_test, X_test], axis=1)
        regressor = RandomForestRegressor(n_estimators=NUM_TREES, n_jobs=-1)
        regressor.fit(train_data, np.ravel(y))
        predictions = regressor.predict(test_data)
        r2 = r2_score(test_Y[gene], predictions)
        pd.DataFrame({'r2': [r2]}, index=[gene]).to_csv(f"{MODEL_OUTPUT_PATH}/SR/result_{gene}.csv")
        joblib.dump(regressor, f"{MODEL_OUTPUT_PATH}/SR/model/rfrg{gene}.pkl")
        pd.DataFrame(predictions).to_csv(f"{MODEL_OUTPUT_PATH}/SR/predictions/pred{gene}.csv")
