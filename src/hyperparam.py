from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sklearn.neighbors as neighbors
import sklearn.svm as svm
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
import sklearn.preprocessing as preprocessing
import sklearn.neural_network as neural_network
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import numpy as np
import functions
import sklearn.tree as tree
import sklearn.naive_bayes as bayes
import pandas as pd
from sklearn.model_selection import cross_validate, KFold
import random


# seeding
SEED = 2
random.seed(SEED)
np.random.seed(SEED)


def createArrays(file_name):
    src = f'src/files/{file_name}'
    f = open(src, "r")
    xValues = []
    yValues = []
    for line in f:
        arr = line.split("\t")
        x = []
        for i in range(0, len(arr)-1):
            value = arr[i]
            if value == "Absent":
                value = 0
            elif value == "Present":
                value = 1
            else:
                value = float(arr[i])
            x.append(value)
        y = int(arr[len(arr)-1])
        xValues.append(x)
        yValues.append(y)
    f.close()
    return xValues, yValues

"""
# create arrays to use from the data
x1, y1 = functions.createArrays("src/files/set1.txt") 
x2, y2 = functions.createArrays("src/files/set2.txt")

print(compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y1))
print(compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y2))
"""
params_dt1 = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],

    'min_samples_leaf': [1, 5, 10],
    'max_leaf_nodes': [None, 5, 10, 15],

    'splitter': ['best', 'random']

}
params_bayes1 = {

    'var_smoothing': [1e-10, 1e-9, 1e-8]

}
params_knn1 = {
    'kneighborsclassifier__n_neighbors': [3, 5, 7, 10],
    'kneighborsclassifier__weights': ['uniform', 'distance']}
params_svm1 = {

    'svc__C': [0.01, 0.1, 1, 10],

    'svc__kernel': ['linear', 'rbf']
}
params_nn1 = {
    'mlpclassifier__hidden_layer_sizes': [(200,100), (100,90), (10, 60)],

    'mlpclassifier__alpha': [0.0001, 0.05],

    'mlpclassifier__learning_rate': ['constant', 'adaptive'],
    'mlpclassifier__solver': ['sgd', 'adam'],
}

params_dt2 = {
    'max_depth': 5,
    'min_samples_split': 2,

    'min_samples_leaf': 1,
    'max_leaf_nodes': 5,

    'splitter': 'best'

}
params_bayes2 = {

    'var_smoothing': 1e-10

}
params_knn2 = {
    'kneighborsclassifier__n_neighbors': 5,
    'kneighborsclassifier__weights': 'uniform'
}
params_svm2 = {

    'svc__C': 1,

    'svc__kernel': 'linear'
}
params_nn2 = {
    'mlpclassifier__hidden_layer_sizes': (200,100),

    'mlpclassifier__alpha': 0.05,

    'mlpclassifier__learning_rate': 'constant',
    'mlpclassifier__solver': 'adam',
}


params_dt3 = {
    'max_depth': 10,
    'min_samples_split': 2,

    'min_samples_leaf': 5,
    'max_leaf_nodes': 10,

    'splitter': 'random'

}
params_bayes3 = {

    'var_smoothing': 1e-9

}
params_knn3 = {
    'kneighborsclassifier__n_neighbors': 10,
    'kneighborsclassifier__weights': 'distance'
}
params_svm3 = {

    'svc__C': 2,

    'svc__kernel': 'linear'
}
params_nn3 = {
    'mlpclassifier__hidden_layer_sizes': (100,90),

    'mlpclassifier__alpha': 0.0001,

    'mlpclassifier__learning_rate': 'constant',
    'mlpclassifier__solver': 'adam',
}


params_dt4 = {
    'max_depth': 2,
    'min_samples_split': 5,

    'min_samples_leaf': 3,
    'max_leaf_nodes': 10,

    'splitter': 'random'

}
params_bayes4 = {

    'var_smoothing': 1e-8

}
params_knn4 = {
    'kneighborsclassifier__n_neighbors': 3,
    'kneighborsclassifier__weights': 'distance'
}
params_svm4 = {

    'svc__C': 3,

    'svc__kernel': 'linear'
}
params_nn4 = {
    'mlpclassifier__hidden_layer_sizes': (200,100),

    'mlpclassifier__alpha': 0.0001,

    'mlpclassifier__learning_rate': 'constant',
    'mlpclassifier__solver': 'adam',
}


def hyperparameter_tune_tree(X:np.ndarray, y:np.ndarray,params_dt:dict)->dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    hyp_tree  = GridSearchCV(tree.DecisionTreeClassifier(), params_dt)
    hyp_tree.fit(X_train, y_train)
    #results = {'best_estimate':hyp_tree.best_estimator_, 'best_parmas': hyp_tree.best_params_, }
    return hyp_tree.best_estimator_, hyp_tree.best_params_

def hyperparameter_tune_naive_bayes(X:np.ndarray, y:np.ndarray,params_nb:dict)->dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    hype_bayes = GridSearchCV(bayes.GaussianNB(), params_nb)
    hype_bayes.fit(X_train, y_train)
    return hype_bayes.best_estimator_,hype_bayes.best_params_

def hyperparameter_tune_knn(X:np.ndarray, y:np.ndarray,params_knn:dict)->dict:
    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    pipeline_knn = make_pipeline(preprocessing.StandardScaler(), neighbors.KNeighborsClassifier())
    hyp_knn = GridSearchCV(pipeline_knn, params_knn)
    hyp_knn.fit(X_train, y_train)
    return hyp_knn.best_estimator_, hyp_knn.best_params_

def hyperparameter_tune_svm(X:np.ndarray, y:np.ndarray,params_svm:dict)->dict:
    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    pipeline_svm = make_pipeline(preprocessing.StandardScaler(), svm.SVC()) 
    hyp_svm = GridSearchCV(pipeline_svm, params_svm)
    hyp_svm.fit(X_train, y_train)
    return hyp_svm.best_estimator_,hyp_svm.best_params_

def hyperparameter_tune_nn(X:np.ndarray, y:np.ndarray, params_nn:dict)->dict:
    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    pipeline_nn = make_pipeline(preprocessing.StandardScaler(), neural_network.MLPClassifier(max_iter=500, early_stopping=True))
    hyp_nn = GridSearchCV(pipeline_nn, params_nn)
    hyp_nn.fit(X_train, y_train)
    return hyp_nn.best_estimator_,hyp_nn.best_params_

def run_everything(X:np.ndarray, y:np.ndarray, params_dt: dict, params_nb: dict, params_knn: dict, params_svm: dict, params_nn: dict):
    pass

def applyCrossValidation(model, X, y):
    kf = KFold(n_splits=10, shuffle=False)
    scores = cross_validate(model, X, y, cv = kf, scoring=['accuracy', 'precision', 'recall', 'f1'])
    results = {

        'mean accuracy': scores['test_accuracy'].mean(),
        'mean precision': scores['test_precision'].mean(),
        'mean recall': scores['test_recall'].mean(),
        'mean f1 score': scores['test_f1'].mean()
    }
    return results

def hyperparamter_tune_save(X, y, output_file_name:str, params_dt, params_bayes, params_knn, params_svm, params_nn):
    results = []

    dt_model, dt_params = hyperparameter_tune_tree(X, y, params_dt)
    dt_cv = applyCrossValidation(dt_model, X, y)
    results.append({'Model': 'Decision Tree', 'Params': dt_params, **dt_cv})

    bayes_model, bayes_params = hyperparameter_tune_naive_bayes(X, y, params_bayes)
    bayes_cv = applyCrossValidation(bayes_model, X, y)
    results.append({'Model': 'Naive Bayes', 'Params': bayes_params, **bayes_cv})

    knn_model, knn_params = hyperparameter_tune_knn(X, y, params_knn)
    knn_cv = applyCrossValidation(knn_model, X, y)
    results.append({'Model': 'KNN', 'Params': knn_params, **knn_cv})

    svm_model, svm_params = hyperparameter_tune_svm(X, y, params_svm)
    svm_cv = applyCrossValidation(svm_model, X, y)
    results.append({'Model': 'SVM', 'Params': svm_params, **svm_cv})

    nn_model, nn_params = hyperparameter_tune_nn(X, y, params_nn)
    nn_cv = applyCrossValidation(nn_model, X, y)
    results.append({'Model': 'NN', 'Params': nn_params, **nn_cv})

    df = pd.DataFrame(results)
    df.to_csv(output_file_name, index=False)

def hyperparamter_tune_custom(X, y, output_file_name, params_dt, params_bayes, params_knn, params_svm, params_nn):
    results = []

    treeModel = tree.DecisionTreeClassifier(max_depth=params_dt["max_depth"], min_samples_split=params_dt["min_samples_split"], min_samples_leaf=params_dt["min_samples_leaf"], max_leaf_nodes=params_dt["max_leaf_nodes"], splitter=params_dt["splitter"])
    dt_cv = applyCrossValidation(treeModel, X, y)
    results.append({'Model': 'Decision Tree', **dt_cv})

    bayes_model = bayes.GaussianNB(var_smoothing=params_bayes["var_smoothing"])
    b_cv = applyCrossValidation(bayes_model, X, y)
    results.append({'Model': 'Bayes', **b_cv})

    knn_model = make_pipeline(preprocessing.StandardScaler(), neighbors.KNeighborsClassifier(
        n_neighbors=params_knn["kneighborsclassifier__n_neighbors"],
        weights=params_knn["kneighborsclassifier__weights"]
        )) 
    knn_cv = applyCrossValidation(knn_model, X, y)
    results.append({'Model': 'KNN', **knn_cv})  

    svm_model = make_pipeline(preprocessing.StandardScaler(), svm.SVC(
        C=params_svm['svc__C'],
        kernel=params_svm["svc__kernel"]
    )) 
    svc_cv = applyCrossValidation(svm_model, X, y)
    results.append({'Model': 'SVM', **svc_cv})  

    nn_model = make_pipeline(preprocessing.StandardScaler(), neural_network.MLPClassifier(
        hidden_layer_sizes=params_nn["mlpclassifier__hidden_layer_sizes"],
        alpha=params_nn["mlpclassifier__alpha"],
        learning_rate=params_nn["mlpclassifier__learning_rate"],
        solver=params_nn["mlpclassifier__solver"]
    )) 
    nn_cv = applyCrossValidation(nn_model, X, y)
    results.append({'Model': 'NN', **nn_cv})  

    df = pd.DataFrame(results)
    df.to_csv(output_file_name, index=False)


# X, y= createArrays(src='src/files/set2.txt') #/Users/prestonstuff/Documents/GitHub/project1b/project1_dataset2.txt


# hyperparamter_tune_save(X, y)


def optimized(file_name, output_file_name):
    X, y = createArrays(file_name=file_name)
    hyperparamter_tune_save(X, y, output_file_name, params_dt=params_dt1, params_bayes = params_bayes1, params_knn = params_knn1, params_svm = params_svm1, params_nn = params_nn1)

def tune1(file_name, output_file_name):
    X, y = createArrays(file_name=file_name)
    hyperparamter_tune_custom(X, y, output_file_name, params_dt=params_dt2, params_bayes = params_bayes2, params_knn = params_knn2, params_svm = params_svm2, params_nn = params_nn2)

def tune2(file_name, output_file_name):
    X, y = createArrays(file_name=file_name)
    hyperparamter_tune_custom(X, y, output_file_name, params_dt=params_dt3, params_bayes = params_bayes3, params_knn = params_knn3, params_svm = params_svm3, params_nn = params_nn3)

def tune3(file_name, output_file_name):
    X, y = createArrays(file_name=file_name)
    hyperparamter_tune_custom(X, y, output_file_name, params_dt=params_dt4, params_bayes = params_bayes4, params_knn = params_knn4, params_svm = params_svm4, params_nn = params_nn4)
