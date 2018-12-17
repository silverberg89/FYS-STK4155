"""
SVM and logistic regression with cross validation
"""

import numpy as np
import matplotlib.pylab as plt
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from to_binary import to_binary
import metrics
from k_fold_CV import k_fold_CV
from Data import import_election_data
    
class Classification:
    
    def __init__(self, k = 0, CV = False, lambdas = [], model = 'Linear SVM'):
        
        """
        k: k-fold batch size
        CV: Cross validation True or False.
        lambdas: In Linear SVM, soft margins. In logistic regression, a list of different lambda values applied. 
        model: Linear SVM, Kernel SVM or Logistic regression.
        """
        
        # Parameters
        self.model = model
        self.k = k
        self.CV = CV
        self.n = len(lambdas)
        self.lambdas = lambdas 
  
    def train_and_crossvalidate(self, X, y):
        
        """
        Train and crossvalidate the model selected.
        X: The design matrix or the explainable variables.
        y: The response variable. 
        """
        
        # Variables
        Xm, Xn = np.shape(X)
        self.Accuracy_train = np.zeros((self.n,))
        self.Accuracy_train_CV = np.zeros((self.n,))
        self.coef = {}  
    
        for l in range(self.n):
        
            # Create model
            if self.model == 'Linear SVM':
                clf = LinearSVC(C = self.lambdas[l], fit_intercept = False)
            if self.model == 'Kernel SVM':    
                clf = SVC(C = self.lambdas[l], kernel = 'rbf', gamma = 'auto')
            if self.model == 'Logistic regression':    
                clf = LogisticRegression(penalty = 'l2', C = 1/self.lambdas[l], fit_intercept = False, solver = 'liblinear')
            
            # Calculation of coefficients
            clf.fit(X, y)
            
            # Calculation of predictions 
            yhat = clf.predict(X)
            yhat = to_binary(yhat, 0.5)
                
            # Accuracy calculation
            self.Accuracy_train[l] = metrics.accuracy(y, yhat)
            
            # Store model
            self.coef[l] = clf
           
            if self.CV == True:
                
                # Cross validation k-fold 
                
                # Variables
                CV_pred = np.zeros(())
            
                for X_train, X_test, y_train, y_test in k_fold_CV(self.k, X, y):
                    
                    # Create model 
                    if self.model == 'Linear SVM':
                        clf = LinearSVC(C = self.lambdas[l], fit_intercept = False)
                    if self.model == 'Kernel SVM':    
                        clf = SVC(C = self.lambdas[l], kernel = 'rbf', gamma = 'auto')
                    if self.model == 'Logistic regression':    
                        clf = LogisticRegression(penalty = 'l2', C = 1/self.lambdas[l], fit_intercept = False, solver = 'liblinear')
                    
                    # Calculation of coefficients  
                    clf.fit(X_train, y_train)
                    
                    # Calculation of predictions 
                    yhat = clf.predict(X_test)
                    CV_pred = np.concatenate((CV_pred, yhat), axis = None)
                    
                    # Store model
                    self.coef[l] = clf
                               
                # Accuracy calculation
                CV_pred = to_binary(CV_pred[1:], 0.5) 
                self.Accuracy_train_CV[l] = metrics.accuracy(y, CV_pred)  
                
            # Print progress
            print('Progress:', l+1, '/', self.n)
    
    def predict(self, X_test, y_test):
    
        """
        Test the model selected.
        X_test: The design matrix or the explainable variables.
        y_test: The response variable. 
        """
        
        # Variables 
        Xm, Xn = np.shape(X_test)
        self.Accuracy_test = np.zeros((self.n,))
    
        for l in range(self.n):
        
            # Calculation of predictions 
            yhat = self.coef[l].predict(X_test)
            yhat = np.reshape(yhat, (Xm,))
            yhat = to_binary(yhat, 0.5)
                
            # Accuracy calculation
            self.Accuracy_test[l] = metrics.accuracy(y_test, yhat)  

    def plot_accuracy(self, Accuracy, title):
        
        """
        Plot accuracy.
        """
        
        # Plot accuracy 
        plt.plot(range(self.n), Accuracy,'b-')
        plt.title(self.model + ': ' + title)
        plt.xlabel('Regularization parameter indices')
        plt.ylabel('Accuracy')
        plt.show()


# Import data

# 2012 US election
X_train, X_test, y_train, y_test = import_election_data(2012)

# 2016 US election
X, y = import_election_data(2016)


"""
Linear SVM
"""

# Parameters 
n = 10
C = C = np.linspace(0.001, 30, n, endpoint = True)
CV = True
k = 100 # # CV-fold batch size

# Train and test on the 2012 US election 
SVM = Classification(k, CV, C, model = 'Linear SVM')
SVM.train_and_crossvalidate(X_train, y_train)
SVM.plot_accuracy(SVM.Accuracy_train, 'Accuracy as a function of different C indices. Training data. 2012 election.')
SVM.predict(X_test, y_test)
SVM.plot_accuracy(SVM.Accuracy_test, 'Accuracy as a function of different C indices. Test data 2012. election')

# Test on the 2016 US election
SVM.predict(X, y)
SVM.plot_accuracy(SVM.Accuracy_test, 'Accuracy as a function of different C indices. Testing on the 2016 election.')


"""
SVM kernel
"""

# Parameters 
n = 10
C = np.linspace(0.001, 30, n, endpoint = True)
CV = True
k = 100 # CV-fold batch size

# Train and test on the 2012 US election 
SVM_2 = Classification(k, CV, C, model = 'Kernel SVM')
SVM_2.train_and_crossvalidate(X_train, y_train)
SVM_2.plot_accuracy(SVM_2.Accuracy_train, 'Accuracy train as a function of different C indices. Training data. 2012 election.')
SVM_2.predict(X_test, y_test)
SVM_2.plot_accuracy(SVM_2.Accuracy_test, 'Accuracy test as a function of different C indices. Test data. 2012 election.')

# 2016 US election
SVM_2.predict(X, y)
SVM_2.plot_accuracy(SVM_2.Accuracy_test, 'Accuracy as a function of different C indices. Testing on the 2016 election.')


"""
Logistic regression
"""

# Parameters 
n = 10
CV = True
k = 100 # CV-fold batch size
lambdas = np.logspace(-4, 5, n)

# Train and test on the 2012 US election
Log = Classification(k, CV, lambdas, model = 'Logistic regression')
Log.train_and_crossvalidate(X_train, y_train)
Log.plot_accuracy(Log.Accuracy_train, 'Accuracy train as function of different lambdas indices. Training data. 2012 election.')
Log.predict(X_test, y_test)
Log.plot_accuracy(Log.Accuracy_test, 'Accuracy test as function of different lambdas indices. Test data. 2012 election.')

# Test on the 2016 US election
Log.predict(X, y)
Log.plot_accuracy(Log.Accuracy_test, 'Accuracy new data as function of different lambdas indices. Testing on the 2016 election.')


"""
Where are the classifications incorrect?
"""

best_model_index = 1 
yhat = SVM_2.coef[best_model_index].predict(X)
n = len(yhat)
yhat = np.reshape(yhat, (n,))
yhat = to_binary(yhat, 0.5)
wrong_classified_counties = np.where(y == yhat, 0, 1)
#f = open("wrong_classified_counties.txt", "w+")
#for i in range(n):
#     f.write('%d \n' % wrong_classified_counties[i])
#f.close() 