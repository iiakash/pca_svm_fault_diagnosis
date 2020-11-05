#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

#Importing the Dataset
dataset_training = pd.read_csv('d00d01.csv') #Please import the training data
dataset_testing = pd.read_csv('d01_te.dat.csv') #Please import the testing data
X_train = dataset_training.iloc[:, 0:52].values
Y_train = dataset_training.iloc[:, 52].values
X_test = dataset_testing.iloc[:, 0:52].values
Y_test = dataset_testing.iloc[:, 52].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

'''#Feature Selection
estimator = SVR(kernel = 'linear')
selector = RFE(estimator, 30, step = 1)
selector1 = selector.fit(X_train, Y_train)
selector2 = selector.fit(X_test, Y_test)
selector1.support_
selector1.ranking_
selector2.support_
selector2.ranking_
X_optTrain = selector1.transform(X_train)
X_optTest = selector2.transform(X_test)'''


#Applying PCA #Please turn off when applying KPCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 37)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

#Explained variance
pca = PCA().fit(X_train)
plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker = 'o', markerfacecolor = 'blue', markersize = 12, color = 'blue', linewidth = 4)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

'''#Applying Kernel PCA #Please Turn Off when applying PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components =32, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)'''

'''#Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 35)
X_train = lda.fit_transform(X_train, Y_train)
X_test = lda.fit_transform(X_test, Y_test)'''

#Fitting SVM to the Training Set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0) #kernel can be changed to linear for linear SVM
classifier.fit(X_train, Y_train)

'''#Fitting Decision Tree to the Training Set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)'''

#Predicting the Test Set Results
Y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

'''#Applying K-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10, n_jobs = -1)
accuracies.mean()
accuracies.std()'''

'''#Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']}, {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.3, 0.1, 0.01, 0.001, 0.0001]}]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_'''

#Visualization
SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 20
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.figure(figsize=(15,10))
plt.plot(Y_pred, color = 'blue')
plt.xlabel('Number of samples')
plt.ylabel('Action')
plt.show

#Fault Diagnosis #Recommended to restart kernel for avoiding misinterpretation
#Applying Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

DatasetRFE = pd.read_csv('d00d19_te.dat.csv')
X = DatasetRFE.iloc[:, 0:52].values
Y = DatasetRFE.iloc[:, 52].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)

estimator = SVR(kernel = 'linear')
selector = RFE(estimator, 10, step = 1)
selector = selector.fit(X, Y)
selector.support_
selector.ranking_
'''X_opt = selector.transform(X)''' #optimum set of features

'''#Applying Backward Elimination
DatasetBE = pd.read_csv('d13_te.dat.csv')
X = DatasetBE.iloc[:, 0:52].values
Y = DatasetBE.iloc[:, 52].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
