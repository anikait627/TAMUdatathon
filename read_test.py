import pandas as pd
import numpy as np
import sklearn as sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from matplotlib import pyplot as plt
​
# cleans training data
def clean_data(training):
    # change na's to 0
    replace_na = df.replace('na',0)
    return replace_na
​
​
def read_file(filename):
    raw_df = pd.read_csv(filename)
    return raw_df
​
def split_response(raw_df):
    # get target vector
    response = raw_df['target']
​
    # get df of training data
    training = raw_df.drop(columns="target")
    
    return response, training
​
# our simplest regression method (Logistical Regression)
def logistic_regression(training,response):
    X = training.drop(columns="id")
    y = response
    
    model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X, y)
    return model
​
#Random Forest model
def RandomForest(training,response):
    X = training.drop(columns="id")
    y = response
    RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0).fit(X, y)
​
    return RF
​
​
#support vector machines
def vector_machine(training,response):
    X = training.drop(columns="id")
    Y = response
​
    SVM = svm.LinearSVC()
    SVM.fit(X,Y)
​
    return round(SVM.score(X,Y),4)
    #SVM.predict(X.iloc[2:,:])
​
def produce_prediction_vector(model, test_data):
    result =model.predict(test_data)
    # print("length:", len(result))
    print("length:", len(result))
    # test_data = pd.DataFrame(result.T)
    final_str = "id,target\n"
    for i, r in enumerate(result):
        final_str += str(i+1) + "," + str(r) + "\n"
    
    # test_data.to_csv("result2.csv", header=False, index=True)
    # f = open("result4.csv", "w+")
    # f.write(final_str)
    # f.close()
​
    
    
if __name__ == "__main__":
    # read in training set
    filename = "equip_failures_training_set.csv"
    df = read_file(filename) 
    response, training = split_response(df) # for now training contains ID column, but is dropped when training 
​
    # clean training data
    training = clean_data(training)
​
    # generate logistic regression model
    model = logistic_regression(training,response)
​
    #RF models = RandomForest(response, training)
    #RF = 
    # test LR against training data
    # print(model.score(training.drop(columns="id"), response))
​
    # #test SVM model against training data
    # svm_model = vector_machine(training,response)
    # print("\nsvm model: ", svm_model)
​
    # test LR model on test file and put resulting vector into csv
    filename = "conoco_test.csv"
​
    #get length for csv
​
    test_df = read_file(filename)
    test_df = clean_data(test_df)
​
​
    produce_prediction_vector(model, test_df.drop(columns="id"))