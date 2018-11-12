import numpy as np
import logmethods
import regressions
np.random.seed(12)

class Linear_reg():
    ''' Sets up linear regression methods ols, ridge, lasso
        and call the calculations and measurements'''
    def __init__(
        self,
        Y_train,
        X_test,
        X_train,
        Y_test):
        self.Y_train = Y_train
        self.X_test = X_test
        self.X_train = X_train
        self.Y_test = Y_test
        
    def ols(self):
        self.yp_ols,self.J_ols   = regressions.ols(self.X_train,self.Y_train,self.X_test)
        self.mse_ols        = logmethods.mse(self.Y_test,self.yp_ols)
        self.r2_ols         = logmethods.r2(self.Y_test,self.yp_ols)
        self.bias2_ols      = logmethods.bias2(self.Y_test,self.yp_ols)
        self.variance_ols   = logmethods.variance(self.yp_ols)
        
    def ridge(self,pen_r):
        self.yp_rid,self.J_rid   = regressions.ridge(self.X_train,self.Y_train,self.X_test,pen_r)
        self.mse_rid        = logmethods.mse(self.Y_test,self.yp_rid)
        self.r2_rid         = logmethods.r2(self.Y_test,self.yp_rid)
        self.bias2_rid      = logmethods.bias2(self.Y_test,self.yp_rid)
        self.variance_rid   = logmethods.variance(self.yp_rid)
        
    def lasso(self,pen_l):
        self.yp_las,self.J_las   = regressions.lasso(self.X_train,self.Y_train,self.X_test,pen_l)
        self.mse_las        = logmethods.mse(self.Y_test,self.yp_las)
        self.r2_las         = logmethods.r2(self.Y_test,self.yp_las)
        self.bias2_las      = logmethods.bias2(self.Y_test,self.yp_las)
        self.variance_las   = logmethods.variance(self.yp_las)