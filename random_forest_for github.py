#Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import pickle
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
print('Libraries Imported')



train_data = pd.read_csv(r'C:\Modelling\train_count.csv')
test_data = pd.read_csv(r'C:\\Modelling\test_count.csv')
train_data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)

#Creating the dependent variable class
factor = pd.factorize(train_data['Classification'])
train_data.Classification = factor[0]
definitions = factor[1]

#Splitting the data into independent and dependent variables
X_train = train_data.iloc[:,12:].values
y_train = train_data.iloc[:,10].values
X_test = test_data.iloc[:,12:].values
y_test = test_data.iloc[:,10].values

# # Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 1800, criterion = 'gini')
forest = classifier.fit(X_train, y_train)


# from pprint import pprint
# # Look at parameters used by our current forest
# print('Parameters currently in use:\n')
# pprint(classifier.get_params())

# print(y_test[:5])


# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Reverse factorize -> remove this if u dont need. instead of 4, i tag as cat 4.
reversefactor = dict(zip(range(4),definitions))
#y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)
test_data['Predicted Category'] = y_pred.tolist()


# df_confusion = pd.crosstab(y_test, y_pred, rownames=['Actual Category'], colnames=['Predicted Category'])
# print(df_confusion)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
#acc = accuracy_score(y_test, y_pred, normalize= False) Normalize = False returns number of accurate predictions 
print("Accuracy of training data is " + str(acc))

#storing the trained model using pickle
# filename = 'final_random_forest_model.sav'
# pickle.dump(classifier, open(filename,'wb'))
# loaded_model = pickle.load(open('final_random_forest_model.sav', 'rb'))
# rf = loaded_model


#or store model using joblib
joblib.dump(classifier, 'randomforestmodel.pkl') 

#loading model using joblib
rf = joblib.load('randomforestmodel.pkl')

df_predict = pd.read_csv(r'C:\DOS Internship\Modelling\base_run_this.csv', low_memory=False)
print(len(df_predict))

X_train = df_predict[df_predict.columns[11:]]

results = rf.predict(X_train)
results2 = np.vectorize(reversefactor.get)(results) # this is a numpy array

#print(type(results2))
df_predict.insert(loc=10, column='Predicted Category', value=results2.tolist())

df_predict.to_csv("say hi to final product.csv")


#feature importance, code might not work in function. copy out and run.
def feature_importance(self):
    # Get numerical feature importances
    importances = list(classifier.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    [print('Variable: {:10} Importance: {}'.format(*pair)) for pair in feature_importances]

#printing out trees of rf
# beware. if n_estimates = 1000, you'll have 1000 .dot files stores 
# if graphviz doesnt show the tree, import online to see/use text editor w inbuilt functions to read .dot files
def tree_print(self):
    # Import tools needed for visualization
    from sklearn.tree import export_graphviz
    import pydot
    import graphviz
    from graphviz import Source
    import os

    # Pull out one tree from the forest
    #tree_in_forest = classifier.estimators_[5]


    #prints out all the trees -- IT WORKS
    i_tree = 0
    for tree_in_forest in forest.estimators_:
        with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
            my_file = tree.export_graphviz(tree_in_forest, out_file = my_file, feature_names = feature_list)
        i_tree = i_tree + 1

# hyper parameter tuning -> no significant increase in accuracy. fyi
def parameter_tuning(self):
    #random hyperparameter grid

    from sklearn.model_selection import RandomizedSearchCV
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 50, num = 5)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    #random_state = [21]
    criterion = ['gini', 'entropy']

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap,
                #'random_state': random_state,
                'criterion' : criterion}
    #pprint(random_grid)

    # First create the base model to tune
    rf = RandomForestClassifier()
    # or, choose 1
    # rf = RandomForestClassifier(n_estimators = 1800, criterion = 'gini')
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=21, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)

    best_random = rf_random.best_estimator_ #this is the best model from randomgrid
    best_random.fit(X_train, y_train)

    y_pred2 = best_random.predict(X_test)
    y_pred2 = np.vectorize(reversefactor.get)(y_pred2)
    # print(y_pred)
    # print(y_test)
    cm2 = confusion_matrix(y_test, y_pred2)
    print(cm2)
    acc2 = accuracy_score(y_test, y_pred2)
    print("Accuracy of training data is " + str(acc2))

    #Grid Search with Cross Validation
    from sklearn.model_selection import GridSearchCV
    # Create the parameter grid based on the results of random search 
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'bootstrap': [True, False],
        'max_depth': [40, 50, 60, 70],
        #'max_features': [2, 3],
        'min_samples_leaf': [1,2],
        'min_samples_split': [1, 2, 3],
        'n_estimators': [1000, 1200, 1400, 1600]
    }
    # Create a based model
    rf = RandomForestClassifier()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                            cv = 3, n_jobs = -1, verbose = 2)


    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    grid_search.best_params_
    best_grid = grid_search.best_estimator_ 


    best_grid.fit(X_train, y_train)

    y_pred3 = best_grid.predict(X_test)
    y_pred3 = np.vectorize(reversefactor.get)(y_pred3)
    # print(y_pred)
    # print(y_test)
    cm3 = confusion_matrix(y_test, y_pred3)
    print(cm3)
    acc3 = accuracy_score(y_test, y_pred3)
    print("Accuracy of training data is " + str(acc3))

    grid_search.best_params_
