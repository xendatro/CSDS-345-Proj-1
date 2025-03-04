Authors: Preston DeLeo and Ethan Ho
Case IDs: pcd42, erh101

# Instructions for Running Project 1

Description on how to run code below:
- cd to root folder
- create venv --> python -m venv env
- *activate venv for windows --> env/Scripts/activate 
- *activate venv for mac --> source env/bin/activate
- install dependencies --> pip install -r requirements.txt
- run main.py --> python src/main.py

## Explanation of Code and Outputs
  Two csv files should be generated in root folder (evaluation1.csv, evaluation2.csv) after some time. These files correspond to running the classifiers on the two datasets (project1_dataset1.txt and project1_dataset2.txt). Within the evaluation csv files, it shows the metrics of these classifiers on the given dataset. It shos results for decision tree, naive bayes, KNN, SVM, and neural network classifiers. Each model in the csv gives its best hyperparamters used from hyperparameter tuning (grid/exhaustive search) in the column "Params". The metrics that are provided are accuracy, precision, recall, and f-1 scores. These metrics come from the mean rather than from per-fold. This provide a more accurate and robust estimate of these models' performance. The cross validation is 10-fold as specified in the requirements. 

## Description of Files

1. main.py acts as a main file to run all of the experiments and hyperparameter tuning

2. hyperparam.py is our file for all of our classifiers and hyperparameter tuning methods. The specified hyperparamter values for all experiments are contained here. 

3. The optimized csv's are the metrics and hyperparameters generated from hyperparameter tuning

4. tune1 csv's correspond to the metrics in the first set of experimentation. tune2 csv's correspond to the metrics in the second set of experimentation. tune3 csv's correspond to the metrics in the third set of experimentation.

