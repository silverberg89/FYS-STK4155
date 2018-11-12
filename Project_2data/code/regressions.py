import numpy as np
import logmethods
from sklearn import linear_model
from sklearn.linear_model import Ridge, Lasso, LinearRegression
np.random.seed(12)

def OLS_code(X,z,Xnew):
    ''' Ordinary linear regression code'''
    ''' Handles input data and output is prediction and coefficents'''
    betahat = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(z))
    zpredict = Xnew.dot(betahat)                       
    return(zpredict,betahat)

def Ridge_code(X,z,Xnew,landa_R):
    ''' Ordinary linear regression with code'''
    ''' Handles input data and output is prediction and coefficents'''
    p = X.shape[1]
    betahat_R = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + landa_R*np.eye(p)), X.T), z)
    zpredict_R = Xnew.dot(betahat_R)
    return(zpredict_R,betahat_R)
    
def ols(X,z,Xnew):
    ''' Ordinary linear regression with SKlearn'''
    ''' Handles input data and output is prediction and coefficents'''
    reg=LinearRegression()
    reg.fit(X, z)
    zpredict_SK=reg.predict(Xnew)
    betahat_SK = reg.coef_
    return(zpredict_SK,betahat_SK)
    
def ridge(X,z,Xnew,landa_R):
    ''' Ordinary linear regression with SKlearn'''
    ''' Handles input data and output is prediction and coefficents'''
    reg_R = Ridge(alpha=landa_R)
    reg_R.fit(X, z)
    zpredict_SK_R = reg_R.predict(Xnew)
    betahat_SK_R = reg_R.coef_
    return(zpredict_SK_R,betahat_SK_R.T)
    
def lasso(X,z,Xnew,landa_L):
    ''' Lasso regression with SKlearn'''
    ''' Handles input data and output is prediction and coefficents'''
    stop = 200000
    reg_L = Lasso(alpha=landa_L, max_iter=stop)
    reg_L.fit(X, z) 
    zpredict_SK_L = reg_L.predict(Xnew)
    betahat_SK_L = reg_L.coef_
    return(zpredict_SK_L,betahat_SK_L.T)

def logistic(x_train,y_train,batch_size,epochs,l_rate,reg_term):
    ''' Take in data and parameters for Logistic regression
        with sigmoid as activation function and cross entropy as cost function.
        Uses stochastic gradient decent with mini batches
        gives weights and biases as output.'''
    loss_epoch_history = np.zeros(epochs)
    W = np.random.randn(len(x_train[0]))  
    B = 0.001
    for i in range(epochs):                         
        batchlist_x, batchlist_y, batches = logmethods.batch(x_train,y_train,batch_size)
        loss_epoch = np.zeros(batches)
        for j in range(batches):                  
            batch_x       = np.asarray(batchlist_x[j])
            batch_y       = np.asarray(batchlist_y[j])
            yp            = logmethods.sig_ord(np.dot(batch_x,W) + B) 
            error         = yp-batch_y
            gradient      = np.dot((batch_x.T),error) 
            gradient     += reg_term*W
            GB = sum(error)
            W             = W - l_rate * gradient #/ batch_size 
            B             = B - l_rate * GB #/ batch_size 
            loss_epoch[j] = np.sum(np.ma.masked_invalid((-batch_y*np.log(yp)-(1-batch_y)*np.log(1-yp))))
        loss_epoch_history[i] = np.mean(loss_epoch)      
        print('Epoch: ',i,'Cost: ',loss_epoch_history[i])
    return(W,B)
    
def logistic_sk(X_test,Y_test,X_train,Y_train,epochs,l_rate,reg_term):
    ''' Logistisc regression with by Sklearn '''
    ''' Handles input data and output is accuracy score for train and test data'''

    logreg_SGD = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=reg_term, max_iter=epochs, 
                                           shuffle=True, random_state=1, learning_rate='optimal')
    logreg_SGD.fit(X_train,Y_train)
    W = logreg_SGD.coef_
    W = W.T
    train_accuracy_SGD =logreg_SGD.score(X_train,Y_train)
    test_accuracy_SGD =logreg_SGD.score(X_test,Y_test)
    return(train_accuracy_SGD,test_accuracy_SGD)