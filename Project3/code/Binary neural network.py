"""
Binary neural network
"""

# Required packages
import numpy as np
import tensorflow as tf
from keras import regularizers
import seaborn as sns
import matplotlib.pylab as plt
sns.set()        

from Data import import_election_data
from k_fold_CV import k_fold_CV
import metrics
from to_binary import to_binary
    
class Binary_neural_network:
    
    def __init__(self, CV = True, k = None, etas = [0.01], lambdas = [0.01], epochs = 10, batch_size_1 = None, batch_size_2 = None, n_hidden_neurons = 100):
        
        """
        model: Regression or binary.
        CV: Cross validation True or False.
        k: k-fold batch size.
        etas: The learning rate values.
        lambdas: The regularization parameters.
        epochs = The number of back and forth propagations.
        batch_size_1: Batch size in an epoch.
        batch_size_2: Batch size in an epoch.
        """
        
        # Parameters 
        self.CV = CV
        self.k = k
        self.etas = etas
        self.lambdas = lambdas 
        self.epochs = epochs
        self.batch_size_1 = batch_size_1
        self.batch_size_2 = batch_size_2
        self.n_hidden_neurons = n_hidden_neurons   
        
    def train(self, X, y):
        
        """
        Train and crossvalidate the model selected.
        X: The design matrix or the explainable variables.
        y: The response variable. 
        """
        
        # Variables
        Xm, Xn = np.shape(X)
        n1 = len(self.etas)
        n2 = len(self.lambdas)
        n = n1 * n2
        self.Accuracy_train = np.zeros((n1, n2))
        self.Accuracy_train_CV = np.zeros((n1, n2))
        self.coef = {} 
        i = 0
        c = 0

        for eta in etas:
            
            j = 0
            
            for l in lambdas:
                
                # Tensorflow / Keras 
                
                # Setting up the network
                clf = tf.keras.Sequential()
                clf.add(tf.keras.layers.Dense(self.n_hidden_neurons, activation = 'relu', input_dim = Xn, kernel_regularizer = regularizers.l2(l)))
                clf.add(tf.keras.layers.Dense(self.n_hidden_neurons, activation = 'relu', kernel_regularizer = regularizers.l2(l)))
                clf.add(tf.keras.layers.Dense(self.n_hidden_neurons, activation = 'relu', kernel_regularizer = regularizers.l2(l)))
                clf.add(tf.keras.layers.Dense(self.n_hidden_neurons, activation = 'relu', kernel_regularizer = regularizers.l2(l)))
                clf.add(tf.keras.layers.Dense(1, activation = 'sigmoid', kernel_regularizer = regularizers.l2(l)))
                sgd = tf.keras.optimizers.SGD(lr = eta)
                clf.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
                
                # Training 
                clf.fit(X, y, epochs = self.epochs, batch_size = self.batch_size_1, verbose = 0)
                
                # Calculation of predictions 
                yhat = np.reshape(clf.predict(X), (Xm,))
                yhat = to_binary(yhat, 0.5)
                
                # Accuracy calculation 
                self.Accuracy_train[i, j] = metrics.accuracy(y, yhat)   
                
                # Store model 
                self.coef[c] = clf
                
                if self.CV == True:
                    
                    # Cross validation k-fold
                    print('CV')
                    
                    # Variables 
                    self.CV_pred = np.zeros(())
                    
                    for X_train, X_test, y_train, y_test in k_fold_CV(self.k, X, y):
        
                        # Variables 
                        X_train_m, X_train_n = np.shape(X_train) 
                        X_test_m, X_test_n = np.shape(X_test) 
                    
                        # Tensorflow / Keras 
                        
                        # Setting up the network
                        clf = tf.keras.Sequential()
                        clf.add(tf.keras.layers.Dense(self.n_hidden_neurons, activation = 'relu', input_dim = Xn, kernel_regularizer = regularizers.l2(l)))
                        clf.add(tf.keras.layers.Dense(self.n_hidden_neurons, activation = 'relu', kernel_regularizer = regularizers.l2(l)))
                        clf.add(tf.keras.layers.Dense(self.n_hidden_neurons, activation = 'relu', kernel_regularizer = regularizers.l2(l)))
                        clf.add(tf.keras.layers.Dense(self.n_hidden_neurons, activation = 'relu', kernel_regularizer = regularizers.l2(l)))
                        clf.add(tf.keras.layers.Dense(1, activation = 'sigmoid', kernel_regularizer = regularizers.l2(l)))
                        sgd = tf.keras.optimizers.SGD(lr = eta)
                        clf.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
                        
                        # Training 
                        clf.fit(X_train, y_train, epochs = self.epochs, batch_size = self.batch_size_2, verbose = 0)
                        
                        # Cross validation predictions
                        yhat = clf.predict(X_test)
                        yhat = np.reshape(yhat, (X_test_m,))
                        self.CV_pred = np.concatenate((self.CV_pred, yhat), axis = None)
        
                    # Cross validation regression statistics calulations   
                    self.CV_pred = self.CV_pred[1:]
                    self.CV_pred = to_binary(self.CV_pred, 0.5)
                    self.Accuracy_train_CV[j, i] = metrics.accuracy(y, self.CV_pred)  
                      
                # Print progress    
                print('Progress:', c+1, '/', n)
                
                j += 1
                c += 1
            i += 1 
    
    def predict(self, X_test, y_test):
        
        """
        X_test: The design matrix or the explainable variables.
        y_test: The response variable. 
        """
        
        # Variables 
        Xm, Xn = np.shape(X_test)
        n1 = len(self.etas)
        n2 = len(self.lambdas)
        self.Accuracy_test = np.zeros((n1, n2))
        c = 0
        
        for i in range(len(self.etas)):
            for j in range(len(self.lambdas)):
        
                # Calculation of predictions
                self.yhat = self.coef[c].predict(X_test)
                self.yhat = np.reshape(self.yhat, (Xm,))
                self.yhat = to_binary(self.yhat, 0.5)
                
                # Accuracy calculation
                self.Accuracy_test[i, j] = metrics.accuracy(y_test, self.yhat)   
            
                c += 1
        
    def heatmap(self, metrics, title):
        
        """
        Accuracy heatmap.
        """
        
        # Heatmap        
        ax = sns.heatmap(metrics, annot = True, cmap = "viridis", yticklabels = self.etas, xticklabels = self.lambdas)
        ax.set_title(title)
        ax.set_ylabel("$\eta$")
        ax.set_xlabel("$\lambda$")
        plt.show()    
        

# Import data
X_train, X_test, y_train, y_test = import_election_data(2012)

# Parameters
CV = True
epochs = 10
k = 500 # CV-fold batch size
etas = [0.01, 0.1, 1]
lambdas = [0.001, 0.01, 0.1]
n_hidden_neurons = 1000
batch_size_1 = 32
batch_size_2 = 32

# Initilize and train the neural network 
Binary = Binary_neural_network(CV, k, etas, lambdas, epochs, batch_size_1, batch_size_2, n_hidden_neurons)
Binary.train(X_train, y_train)
# Model metrics 
Binary.heatmap(Binary.Accuracy_train, "Accuracy train. 2012 election.")

# Test the model on unseen data.  
Binary.predict(X_test, y_test)
Binary.heatmap(Binary.Accuracy_test, "Accuracy test. 2012 election.")

# Test on the 2016 Election
X, y = import_election_data(2016)
Xm, Xn = X.shape
Binary.predict(X, y)
Binary.heatmap(Binary.Accuracy_test, "Accuracy test. 2016 election.")

