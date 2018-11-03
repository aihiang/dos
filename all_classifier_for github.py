#Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn import tree #do sth about this
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier 
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
print('Libraries Imported')

# Reading the dataset, cleaning
dataset = pd.read_csv(r'C:\\Modelling\final_trng_set.csv')
dataset.head()
dataset.fillna(0, inplace=True)

features =dataset.drop('Classification', axis = 1)
feature_list = list(features.columns)

#Creating the dependent variable class
factor = pd.factorize(dataset['Classification'])
dataset.Classification = factor[0]
definitions = factor[1]

#Splitting the data into independent and dependent variables
X = dataset.iloc[:,12:].values
# X = dataset.iloc[:,11:].values # Includes number of pages scraped
y = dataset.iloc[:,10].values

# Creating the Training and Test set from data - Validation set approach, 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 21)


def Gradient_Boost(train_X, test_X, train_y, test_y):
    params = {'n_estimators' : 1800, 'learning_rate' : 0.05}
    clf = GradientBoostingClassifier(**params).fit(train_X, train_y)
    reversefactor = dict(zip(range(4),definitions))
    predicted_y = clf.predict(test_X)
    predicted_y = np.vectorize(reversefactor.get)(predicted_y)
    correct_y = np.vectorize(reversefactor.get)(test_y)
    cm = confusion_matrix(correct_y, predicted_y)
    print('')
    print("THIS IS THE RESULT FOR GRADIENT BOOST")
    print(cm)
    acc = accuracy_score(correct_y, predicted_y)
    print("Accuracy of training data is " + str(acc))
    return acc

def Ada_Boost(train_X, test_X, train_y, test_y):
    params = {'n_estimators' : 1000}
    clf = AdaBoostClassifier(**params).fit(train_X, train_y)
    reversefactor = dict(zip(range(4),definitions))
    predicted_y = clf.predict(test_X)
    predicted_y = np.vectorize(reversefactor.get)(predicted_y)
    correct_y = np.vectorize(reversefactor.get)(test_y)
    cm = confusion_matrix(correct_y, predicted_y)
    print('')
    print("THIS IS THE RESULT FOR ADA_BOOST")
    print(cm)
    acc = accuracy_score(correct_y, predicted_y)
    print("Accuracy of training data is " + str(acc))
    return acc

def Logistic_Regression_CV(train_X, test_X, train_y, test_y):
    clf = LogisticRegressionCV(multi_class='multinomial', solver ='newton-cg', max_iter=4000).fit(train_X, train_y)
    #if no max iter, cannot converge!
    reversefactor = dict(zip(range(4),definitions))
    predicted_y = clf.predict(test_X)
    predicted_y = np.vectorize(reversefactor.get)(predicted_y)
    correct_y = np.vectorize(reversefactor.get)(test_y)
    cm = confusion_matrix(correct_y, predicted_y)
    print('')
    print("THIS IS THE RESULT FOR LOGISTIC REGRESSION CV")
    print(cm)
    acc = accuracy_score(correct_y, predicted_y)
    print("Accuracy of training data is " + str(acc))
    return acc

def Logistic_Regression(train_X, test_X, train_y, test_y):
    # if __name__ == '__main__':
    clf = LogisticRegression(multi_class='multinomial', solver ='newton-cg', max_iter=4000).fit(train_X, train_y)
    #if no max iter, cannot converge!
    # , n_jobs=-1
    reversefactor = dict(zip(range(4),definitions))
    predicted_y = clf.predict(test_X)
    predicted_y = np.vectorize(reversefactor.get)(predicted_y)
    correct_y = np.vectorize(reversefactor.get)(test_y)
    cm = confusion_matrix(correct_y, predicted_y)
    print('')
    print("THIS IS THE RESULT FOR LOGISTIC REGRESSION")
    print(cm)
    acc = accuracy_score(correct_y, predicted_y)
    print("Accuracy of training data is " + str(acc))
    return acc

def SVM(train_X, test_X, train_y, test_y):
    clf = SVC(kernel = 'linear' ).fit(X_train, y_train)
    # kernel = 'sigmoid' # return 0.2
    # kernel = 'poly' # return 0.65
    # kernel = 'rbf' # return 0.38 AKA Gaussian
    reversefactor = dict(zip(range(4),definitions))
    predicted_y = clf.predict(test_X)
    predicted_y = np.vectorize(reversefactor.get)(predicted_y)
    correct_y = np.vectorize(reversefactor.get)(test_y)
    cm = confusion_matrix(correct_y, predicted_y)
    print('')
    print("THIS IS THE RESULT FOR SUPPORT VECTOR MACHINE (SVM)")
    print(cm)
    acc = accuracy_score(correct_y, predicted_y)
    print("Accuracy of training data is " + str(acc))
    return acc

def Neural_Network(train_X, test_X, train_y, test_y):
    clf = MLPClassifier(solver='lbfgs', activation = 'logistic', alpha=1e-5,hidden_layer_sizes=(15, ), random_state=1).fit(train_X, train_y)
    # solver = 'lbfgs' #0.66, use this for smaller trng set, converge faster
    # solver = 'sgd' #0.60
    # solver = 'adam' #0.64
    # activation = 'relu' #0.66
    # activation = 'logistic' #0.71
    # activation = 'tanh' #0.69
    # activation = 'identity' #0.65
    reversefactor = dict(zip(range(4),definitions))
    predicted_y = clf.predict(test_X)
    predicted_y = np.vectorize(reversefactor.get)(predicted_y)
    correct_y = np.vectorize(reversefactor.get)(test_y)
    cm = confusion_matrix(correct_y, predicted_y)
    print('')
    print("THIS IS THE RESULT FOR NEURAL NETWORK")
    print(cm)
    acc = accuracy_score(correct_y, predicted_y)
    print("Accuracy of training data is " + str(acc))
    return acc

def Ridge_Classifier(train_X, test_X, train_y, test_y):
    clf = RidgeClassifier().fit(train_X, train_y)
    reversefactor = dict(zip(range(4),definitions))
    predicted_y = clf.predict(test_X)
    predicted_y = np.vectorize(reversefactor.get)(predicted_y)
    correct_y = np.vectorize(reversefactor.get)(test_y)
    cm = confusion_matrix(correct_y, predicted_y)
    print('')
    print("THIS IS THE RESULT FOR RIDGE CLASSIFIER")
    print(cm)
    acc = accuracy_score(correct_y, predicted_y)
    print("Accuracy of training data is " + str(acc))
    return acc

def Voting_Classifier(all_x, all_y):
    clf1 = LogisticRegression(multi_class='multinomial',solver ='newton-cg', max_iter=4000, random_state=1)
    clf2 = RandomForestClassifier(n_estimators = 1800, criterion = 'gini', random_state=1)
    clf3 = GradientBoostingClassifier(n_estimators = 1800, learning_rate = 0.05 )
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3)], voting='hard') #, n_jobs=-1
    print('')
    print("THIS IS THE RESULT FOR VOTING CLASSIFIER")
    for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'Gradient Boost', 'Ensemble']):
        scores = cross_val_score(clf, all_x, all_y, cv=5, scoring='accuracy') #cross validate compute score 5 consec times
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

def Naive_Bayes(train_X, test_X, train_y, test_y):
    clf = GaussianNB().fit(train_X, train_y)
    reversefactor = dict(zip(range(4),definitions))
    predicted_y = clf.predict(test_X)
    predicted_y = np.vectorize(reversefactor.get)(predicted_y)
    correct_y = np.vectorize(reversefactor.get)(test_y)
    cm = confusion_matrix(correct_y, predicted_y)
    print('')
    print("THIS IS THE RESULT FOR NAIVE BAYES")
    print(cm)
    acc = accuracy_score(correct_y, predicted_y)
    print("Accuracy of training data is " + str(acc))
    return acc

# if __name__ == '__main__':
#     # Logistic_Regression_CV(X_train, X_test, y_train, y_test) # TAKES SUPER DUPER LONG, average 72%
#     Logistic_Regression(X_train, X_test, y_train, y_test) # average 72%
    
# Gradient_Boost(X_train, X_test, y_train, y_test)
# Ada_Boost(X_train, X_test, y_train, y_test) #ADAPTIVE BOOSTING, average 67%
# Logistic_Regression(X_train, X_test, y_train, y_test) # average 72%
# SVM(X_train, X_test, y_train, y_test) #TAKES VERY LONG FOR LINEAR KERNEL, average 68%
# Neural_Network(X_train, X_test, y_train, y_test) #average 71%
# Ridge_Classifier(X_train, X_test, y_train, y_test)
# Voting_Classifier(X, y) #average 77%
# Naive_Bayes(X_train, X_test, y_train, y_test) #average 61%