# Importing packages------------------------------------------------------
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from imageio import imread
from sklearn.model_selection import KFold
import time

#-Functions---------------------------------------------------------------
def FrankeFunction(x,y):
    '''Construct the frankie function'''
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4
    
def res_sq(y, Y):
    '''Squared residual'''
    rsq = sum((y-Y)**2)
    return (rsq)

def MSE(y, Y):
    '''Mean Square Error'''
    n = len(y)
    mse = res_sq(y,Y)*(1 / n)
    return (mse)

def R2(y, Y):
    '''R2-Score'''
    r2 = 1 - (res_sq(y,Y) / sum((y - sum(y) / float(y.size))**2))
    return (r2)

def mean(y, Y):
    '''Absolute mean value'''
    Mean = sum(abs(y-Y)) / float(y.size)
    return (Mean)
    
def var(zhat):
    '''Calculates the variance, zhat = prediction'''
    n = len(z)
    G = np.sum((zhat - np.mean(zhat))**2 )/n
    return(G)
    
def bias2(z,zhat):
    ''' Bias is the measurement of a model to
        over- or under-estimate the value of a population parameter'''
    n = len(zhat)
    Bias = np.sum((z - (np.mean(zhat)))**2)/n
    return(Bias)
    
def bias2_brute(z,zhat):
    ''' Just for checking the above definition'''
    u = 0
    b = 0
    n = len(zhat)
    for i in range(n):
        u+=zhat[i]
    u=u/n
    for k in range(n):
        b += (z[k]-u)**2
    b = b/n
    return(b)
    
def varBetahat(X,G,B):
    ''' Calculates variance of betahat, X = independent data, G = variance
        And calculates the confidence interval of betahat'''
    u = np.dot(X.T,X)
    v = np.linalg.inv(u)*G*G # Covariance matrix
    varB = np.diag(v)        # Var(B) is the diagonal of the covariance matrix
    n = len(varB)
    
    c1 = np.zeros(n)    
    c2 = np.zeros(n)    
    c = 1.96        # Confidence Level: 0.95, Critical value(Z-score): 1.96
    for i in range(n-1):
        c1[i] = B[i]-c*math.sqrt(varB[i])*G  # Left lim of conf interval
        c2[i] = B[i]+c*math.sqrt(varB[i])*G  # Rigth lim of conf interval
    C = np.column_stack((c1,c2))
    
    return(varB,C)
    
def OLS_code(X,z,Xnew):
    ''' Ordinary linear regression code'''
    betahat = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(z))# Identify B parameters
    zpredict = Xnew.dot(betahat)                       # Calculates predicted z
    return(zpredict,betahat)
    
def OLS_SK(X,z,Xnew):
    ''' Ordinary linear regression with SKlearn'''
    reg=LinearRegression()
    reg.fit(X, z)                                      # Identify B parameters 
    zpredict_SK=reg.predict(Xnew)                      # Calculates predicted z
    betahat_SK = reg.coef_
    return(zpredict_SK,betahat_SK.T)
    
def Ridge_code(X,z,Xnew,landa_R):
    ''' Ordinary linear regression with code'''
    p = X.shape[1]
    betahat_R = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + landa_R*np.eye(p)), X.T), z)
    zpredict_R = Xnew.dot(betahat_R)
    return(zpredict_R,betahat_R)
    
def Ridge_SK(X,z,Xnew,landa_R):
    ''' Ordinary linear regression with SKlearn'''
    reg_R = Ridge(alpha=landa_R)
    reg_R.fit(X, z)
    zpredict_SK_R = reg_R.predict(Xnew)
    betahat_SK_R = reg_R.coef_
    return(zpredict_SK_R,betahat_SK_R.T)
    
def Lasso_SK(X,z,Xnew,landa_L,stop):
    ''' Ordinary linear regression with SKlearn'''
    reg_L = Lasso(alpha=landa_L, max_iter=stop)
    reg_L.fit(X, z) 
    zpredict_SK_L = reg_L.predict(Xnew)
    betahat_SK_L = reg_L.coef_
    return(zpredict_SK_L,betahat_SK_L.T)
    
def CrossValidation(z,X,landa_R,landa_L,stop):
    ''' K-fold cross validation method
        Splitt data into training and test sets and draws
        corresponding measurments for mean square error and r2 score.
        Thereafter averaging the measurements.'''
    # Measurment variables
    r2_K = 0
    r2_SK_K = 0
    r2_R_K = 0
    r2_SK_R_K = 0
    r2_SK_L_K = 0
    mse_K = 0
    mse_SK_K = 0
    mse_R_K = 0
    mse_SK_R_K = 0
    mse_SK_L_K = 0
    Va_R_K = 0
    Va_L_K = 0
    Va_O = 0
    bias_R_K = 0
    bias_SK_L_K = 0
    bias_O = 0
    # Split data
    k = 5
    kf = KFold(n_splits=k,shuffle=True)
    fold = 0
    for train_idx, test_idx in kf.split(X):
        fold = fold + 1
        Xtrain, Xtest = X[train_idx], X[test_idx]
        ztrain, ztest = z[train_idx], z[test_idx]
        # Run methods with resampled data
        zpredict_K,betahat_K            = OLS_code(Xtrain,ztrain,Xtest)
        zpredict_SK_K,betahat_SK_K      = OLS_SK(Xtrain,ztrain,Xtest)
        zpredict_R_K,betahat_R_K        = Ridge_code(Xtrain,ztrain,Xtest,landa_R)
        zpredict_SK_R_K,betahat_SK_R_K  = Ridge_SK(Xtrain,ztrain,Xtest,landa_R)
        zpredict_SK_L_K,betahat_SK_L_K  = Lasso_SK(Xtrain,ztrain,Xtest,landa_L,stop)
        # Measurements
        r2_K = r2_K + R2(ztest,zpredict_K)
        r2_SK_K = r2_SK_K + r2_score(ztest,zpredict_SK_K)
        r2_R_K = r2_R_K + R2(ztest,zpredict_R_K)
        r2_SK_R_K = r2_SK_R_K + r2_score(ztest,zpredict_SK_R_K)
        r2_SK_L_K = r2_SK_L_K + r2_score(ztest,zpredict_SK_L_K)
        mse_K = mse_K + MSE(ztest,zpredict_K)
        mse_SK_K = mse_SK_K + mean_squared_error(ztest,zpredict_SK_K)
        mse_R_K = mse_R_K + MSE(ztest,zpredict_R_K)
        mse_SK_R_K = mse_SK_R_K + mean_squared_error(ztest,zpredict_SK_R_K)
        mse_SK_L_K = mse_SK_L_K +mean_squared_error(ztest,zpredict_SK_L_K)
        Va_R_K = var(zpredict_SK_R)
        Va_L_K = var(zpredict_SK_L)
        Va_O = var(zpredict_SK)
        bias_R_K = bias2(ztest,zpredict_R_K)
        bias_SK_L_K = bias2(ztest,zpredict_SK_L_K)
        bias_O = bias2(ztest,zpredict_K)
    # Average Measurments
    Va_R_K = Va_R_K/fold
    Va_L_K = Va_L_K/fold
    Va_O = Va_O/fold
    bias_O = bias_O/fold
    bias_R_K = bias_R_K/fold
    bias_SK_L_K = bias_SK_L_K/fold
    r2_K = r2_K/fold
    r2_SK_K = r2_SK_K/fold
    r2_R_K = r2_R_K/fold
    r2_SK_R_K = r2_SK_R_K/fold
    r2_SK_L_K = r2_SK_L_K/fold
    mse_K = mse_K/fold
    mse_SK_K = mse_SK_K/fold
    mse_R_K = mse_R_K/fold
    mse_SK_R_K = mse_SK_R_K/fold
    mse_SK_L_K = mse_SK_L_K/fold

    return(r2_K,r2_SK_K,r2_R_K,r2_SK_R_K,r2_SK_L_K,mse_K,mse_SK_K,mse_R_K,mse_SK_R_K,mse_SK_L_K,Va_R_K,Va_L_K,bias_R_K,bias_SK_L_K,Va_O,bias_O)

# End of def ----------------------------------------------------------------
run = 6                                         # Determines number of loops-1

# Holders for extracting measurments for each loop
A_variance = np.zeros(run-1)
A_variance_L = np.zeros(run-1)
A_variance_R = np.zeros(run-1)
A_variance_K = np.zeros(run-1)
A_variance_L_K = np.zeros(run-1)
A_variance_R_K= np.zeros(run-1)

A_bias = np.zeros(run-1)
A_bias_L = np.zeros(run-1)
A_bias_R = np.zeros(run-1)
A_bias_K = np.zeros(run-1)
A_bias_L_K = np.zeros(run-1)
A_bias_R_K = np.zeros(run-1)

B_varb_mean = np.zeros(run-1)
B_varb_mean_L = np.zeros(run-1)
B_varb_mean_R = np.zeros(run-1)

R_two = np.zeros(run-1)
R_two_L = np.zeros(run-1)
R_two_R = np.zeros(run-1)
R_two_K = np.zeros(run-1)
R_two_L_K = np.zeros(run-1)
R_two_R_K = np.zeros(run-1)

meanerror = np.zeros(run-1)
meanerror_L = np.zeros(run-1)
meanerror_R = np.zeros(run-1)
meanerror_K = np.zeros(run-1)
meanerror_L_K = np.zeros(run-1)
meanerror_R_K = np.zeros(run-1)

landa_Rid = np.zeros(run-1)
landa_Las = np.zeros(run-1)
amp1 = np.zeros(run-1)

r2_Rid = np.zeros(run-1)
r2_Las = np.zeros(run-1)
r2_OLS = np.zeros(run-1)

mse_Rid = np.zeros(run-1)
mse_Las = np.zeros(run-1)
mse_OLS = np.zeros(run-1)

var_ols = np.zeros(run-1)
var_rid = np.zeros(run-1)
var_las = np.zeros(run-1)

file=open('geo_n60_landa_r=3,landa_l=0.05.txt','a')
#file=open('Landa_R2_deg4, amp0,n200.txt','a')
file.write("First column: degree 1, Last column: degree 5, first row: OSL, second: Lasso, third: Ridge")
file.write("\n-----------------------")

# Start of loop (Change values of deg and landas for analysis)---------------
for ik in range (1,run,1):
    start = time.clock()
    #-Data--------------------------------------------------------------------
    deg = ik                                 # Polynomial degrees
    amp = 0.0                                 # Amplitude of noise
    landa_R = 1                           # Penalty coefficients
    landa_L = 0.005
    n= 20                                  #100 = 383 sec, 150 = 625 (GEodata) # Number of rows
    m= n                                    # Number of columns
    stop = 300000                           # Maximum numer of iterations Lasso
    step = 1/n                              # Step size
    
    np.random.seed(1)                       # Same random set
    noise = amp*np.random.randn(n,1)        # Noice source function
    xp = np.linspace(0,1,n)                 # Create axes
    yp = np.linspace(0,1,m)
    xx, yy = np.meshgrid(xp,yp)             # Meshgrid
    bricks = n*m
    
    #z = FrankeFunction(xx,yy)+noise         # Frankie data
    z = imread('SRTM_data_Norway_1.tif')  # Geo data
    z = z[0:n, 0:m]                       # Cut amount of data
    z = z.flatten()                         # Squeeze down to one array
    
    xpp = np.random.rand(n,1)               # Create random model
    ypp = np.random.rand(m,1)
    xpp = np.sort(xpp, axis=0)
    ypp = np.sort(ypp, axis=0)
    xxx, yyy = np.meshgrid(xpp,ypp)
    x = xxx.flatten()
    y = yyy.flatten()
    ones = np.ones(bricks)
    X = np.c_[  ones,x,y                                 # Poly 1, Col 0-2
           ,x**2,x*y,y**2                                # Poly 2, Col 3-5
           ,x**3,x**2*y,x*y**2,y**3                      # Poly 3, Col 6-9
           ,x**4,x**3*y,x**2*y**2,x*y**3,y**4            # Poly 4, Col 10-14
           ,x**5,x**4*y,x**3*y**2,x**2*y**3,x*y**4,y**5] # Poly 5, Col 15-20
    
    if   deg == 1:                          # Slice to choosen degree
        X = X[:,0:3]
    elif deg == 2:
        X = X[:,0:6]
    elif deg == 3:
        X = X[:,0:10]
    elif deg == 4:
        X = X[:,0:15]
    else:
        X = X[:,0:21]
    
    # Run methods------------------------------------------------------------
    zpredict,betahat            = OLS_code(X,z,X)
    zpredict_SK,betahat_SK      = OLS_SK(X,z,X)
    zpredict_R,betahat_R        = Ridge_code(X,z,X,landa_R)
    zpredict_SK_R,betahat_SK_R  = Ridge_SK(X,z,X,landa_R)
    zpredict_SK_L,betahat_SK_L  = Lasso_SK(X,z,X,landa_L,stop)
    
    # Calculate measurements-------------------------------------------------  
    ''' G(0,1,2,3,4) = Var(OLS,OLS_SK,Ridge,Ridge_SK,Lasso_SK
    Variables was not shown if logical names were assigned i.e (_SK_R..)'''
    mse = MSE(z,zpredict)                 # Mean square error     
    r2 = R2(z,zpredict)                   # Calculates R2 score
    G0 = var(zpredict)                    # Variance by book Eq 3.8
    varb,C0 = varBetahat(X,G0,betahat)    # Var(B), C = conf int av beta
    bias = bias2(z,zpredict)              # Bias 
    
    mse_SK = mean_squared_error(z,zpredict_SK)
    r2_SK = r2_score(z,zpredict_SK)
    G1 = var(zpredict_SK)
    varb_SK,C1 = varBetahat(X,G1,betahat_SK)
    bias_SK = bias2(z,zpredict_SK)
    
    mse_R = MSE(z,zpredict_R)
    r2_R = R2(z,zpredict_R)
    G2 = var(zpredict_R)
    varb_R,C2 = varBetahat(X,G2,betahat_R)
    bias_R = bias2(z,zpredict_R)
    
    mse_SK_R = mean_squared_error(z,zpredict_SK_R)
    r2_SK_R = r2_score(z,zpredict_SK_R)
    G3 = var(zpredict_SK_R)
    varb_SK_R,C3 = varBetahat(X,G3,betahat_SK_R)
    bias_SK_R = bias2(z,zpredict_SK_R)
    
    mse_SK_L = mean_squared_error(z,zpredict_SK_L)
    r2_SK_L = r2_score(z,zpredict_SK_L)
    G4 = var(zpredict_SK_L)
    varb_SK_L,C4 = varBetahat(X,G4,betahat_SK_L)
    bias_SK_L = bias2(z,zpredict_SK_L)
    
    # Resampling------------------------------------------------------------
    '''K-fold cross validation method'''
    r2_K,r2_SK_K,r2_R_K,r2_SK_R_K,r2_SK_L_K,mse_K,mse_SK_K,mse_R_K,mse_SK_R_K,mse_SK_L_K,Va_R_K,Va_L_K,bias_R_K,bias_SK_L_K,Va_O,bias_O = CrossValidation(z,X,landa_R,landa_L,stop)

# =============================================================================
#     # Extract values and measurements for different parameters
#      # Need to comment out if different polys is used
#     landa_Rid[ik-1] = landa_R
#     landa_Las[ik-1] = landa_L
#     amp1[ik-1] = amp
#     
#     r2_Rid[ik-1] = r2_SK_R,6
#     r2_Las[ik-1] = r2_SK_L,6
#     r2_OLS[ik-1] = r2_SK,6
#     
#     mse_Rid[ik-1] = mse_SK_R,8
#     mse_Las[ik-1] = mse_SK_L,8
#     mse_OLS[ik-1] = mse_SK,8
#     
#     var_ols[ik-1] = G1,8
#     var_rid[ik-1] = G3,8
#     var_las[ik-1] = G4,8
# =============================================================================
    # Extract values and measurements for different polys
    # Need to comment out if same poly is used
    A_variance[ik-1] = G0
    A_variance_L[ik-1] = G4
    A_variance_R[ik-1] = G3
    A_variance_K[ik-1] = Va_O
    A_variance_L_K[ik-1] = Va_L_K
    A_variance_R_K[ik-1] = Va_R_K

    A_bias[ik-1] = bias_SK
    A_bias_L[ik-1] = bias_SK_L
    A_bias_R[ik-1] = bias_SK_R
    A_bias_K[ik-1] = bias_O
    A_bias_L_K[ik-1] = bias_SK_L_K
    A_bias_R_K[ik-1] = bias_R_K

    R_two[ik-1] = r2
    R_two_L[ik-1] = r2_SK_L
    R_two_R[ik-1] = r2_R
    R_two_K[ik-1] = r2_K
    R_two_L_K[ik-1] = r2_SK_L_K
    R_two_R_K[ik-1] = r2_R_K

    meanerror[ik-1] = mse
    meanerror_L[ik-1] = mse_SK_L
    meanerror_R[ik-1] = mse_R
    meanerror_K[ik-1] = mse_K
    meanerror_L_K[ik-1] = mse_SK_L_K
    meanerror_R_K[ik-1] = mse_R_K
    
# For plotting the Beta coefficents----------------------------------------
label = [  'ones','x','y'                         
           ,'x**2','x*y','y**2'                      
           ,'x**3','x**2*y','x*y**2','y**3'               
           ,'x**4','x**3*y','x**2*y**2','x*y**3','y**4'     
           ,'x**5','x**4*y','x**3*y**2','x**2*y**3','x*y**4','y**5'] 

index = np.arange(len(label))

fig1 = plt.figure(1)
plt.bar(index, betahat_SK, color='blue', label="OLS")
plt.xlabel('Polynomial type', fontsize=5)
plt.ylabel('Beta Coefficent Value', fontsize=5)
plt.xticks(index, label, fontsize=5, rotation=30)
plt.title('Beta Coefficents')
plt.legend()
plt.show()
fig1.savefig('beta_ols_n200.png', dpi=200)

fig2 = plt.figure(2)
plt.bar(index, betahat_SK_L, color='green', label="Lasso")
plt.xlabel('Polynomial type', fontsize=5)
plt.ylabel('Beta Coefficent Value', fontsize=5)
plt.xticks(index, label, fontsize=5, rotation=30)
plt.title('Beta Coefficents, Landa Lasso = 0.5')
plt.legend()
plt.show()
fig2.savefig('beta_lass_0.5_n200.png', dpi=200)

fig3 = plt.figure(3)
plt.bar(index, betahat_SK_R, color='red', label="Ridge")
plt.xlabel('Polynomial type', fontsize=5)
plt.ylabel('Beta Coefficent Value', fontsize=5)
plt.xticks(index, label, fontsize=5, rotation=30)
plt.title('Beta Coefficents, Landa Ridge 2')
plt.legend()
plt.show()
fig3.savefig('beta_ridge_2_n200.png', dpi=200)
    
# =============================================================================
# # For plotting MSE,R2,VAR for different landa and amplitudes----------------
# fig4 = plt.figure(4)
# plt.plot(landa_Rid,r2_Rid, label="Ridge")
# plt.plot(landa_Rid,r2_OLS, label="OLS")
# plt.xlabel('Landa', fontsize=12)
# plt.ylabel('R2-score', fontsize=12)
# plt.legend()
# plt.title('Landa vs R2-score')
# plt.show()
# fig4.savefig('Landa vs R2-score_Ridge_p5.png')
#  
# fig5 = plt.figure(5)
# plt.plot(landa_Las,r2_Las, label="Lasso")
# plt.plot(landa_Las,r2_OLS, label="OLS")
# plt.xlabel('Landa', fontsize=12)
# plt.ylabel('R2-score', fontsize=12)
# plt.legend()
# plt.title('Landa vs R2-score')
# plt.show()
# fig5.savefig('Landa vs R2-score_Lasso_p5.png')
# 
# fig6 = plt.figure(6)
# plt.plot(landa_Rid,mse_Rid, label="Ridge")
# plt.plot(landa_Rid,mse_OLS, label="OLS")
# plt.xlabel('Landa', fontsize=12)
# plt.ylabel('MSE', fontsize=12)
# plt.legend()
# plt.title('Landa vs R2-score')
# plt.show()
# fig6.savefig('Landa vs MSE_Ridge_p5.png')
#  
# fig7 = plt.figure(7)
# plt.plot(landa_Las,mse_Las, label="Lasso")
# plt.plot(landa_Las,mse_OLS, label="OLS")
# plt.xlabel('Landa', fontsize=12)
# plt.ylabel('MSE', fontsize=12)
# plt.legend()
# plt.title('Landa vs MSE')
# plt.show()
# fig7.savefig('Landa vs MSE_Lasso_p5.png')
# 
# fig8 = plt.figure(8)
# plt.plot(landa_Rid,var_rid, label="Ridge")
# plt.plot(landa_Rid,var_ols, label="OLS")
# plt.xlabel('Landa', fontsize=12)
# plt.ylabel('Var(zpredict)', fontsize=12)
# plt.legend()
# plt.title('Landa vs Variance')
# plt.show()
# fig8.savefig('Landa_vs_Var_ridge_p5.png')
#  
# fig9 = plt.figure(9)
# plt.plot(landa_Las,var_las, label="Lasso")
# plt.plot(landa_Las,var_ols, label="OLS")
# plt.xlabel('Landa', fontsize=12)
# plt.ylabel('Var(zpredict)', fontsize=12)
# plt.legend()
# plt.title('Landa vs Variance')
# plt.show()
# fig9.savefig('Landa_vs_Var_lasso_p5.png')
# 
# fig10 = plt.figure(10)
# plt.plot(amp1,var_rid, label="Ridge")
# plt.plot(amp1,var_ols, label="OLS")
# plt.xlabel('Amp Noise', fontsize=12)
# plt.ylabel('Var(zpredict)', fontsize=12)
# plt.legend()
# plt.title('Landa vs Variance')
# plt.show()
# fig10.savefig('Landa_vs_Amp_ridge_p5.png')
#  
# fig11 = plt.figure(11)
# plt.plot(amp1,var_las, label="Lasso")
# plt.plot(amp1,var_ols, label="OLS")
# plt.xlabel('Amp Noise', fontsize=12)
# plt.ylabel('Var(zpredict)', fontsize=12)
# plt.legend()
# plt.title('Landa vs Variance')
# plt.show()
# fig11.savefig('Landa_vs_Amp_lasso_p5.png')
# 
# =============================================================================
# Plot and save measurments after loop over polynomials----------------------
if ik == 5:
    
# =============================================================================
#     np.savetxt(file,(r2,r2_SK,r2_R,r2_SK_R),fmt='%1.4f',header='R2')
#     file.write("-----------------------")
#     np.savetxt(file,(mse,mse_SK,mse_R,mse_SK_R),fmt='%1.4f',header='MSE')
# =============================================================================
    
# =============================================================================
#     # Save information to file
#     np.savetxt(file,(R_two,R_two_L,R_two_R),fmt='%1.4f',header='R2')
#     file.write("-----------------------")
#     np.savetxt(file,(meanerror,meanerror_L,meanerror_R),fmt='%1.4f',header='MSE')
#     file.write("-----------------------")
#     np.savetxt(file,(A_bias,A_bias_L,A_bias_R),fmt='%1.4f',header='Bias')
#     file.write("-----------------------")
#     np.savetxt(file,(A_variance,A_variance_L,A_variance_R),fmt='%1.4f',header='Var')
#     file.write("-----------------------")
#     np.savetxt(file,(B_varb_mean,B_varb_mean_L,B_varb_mean_R),fmt='%1.4f',header='Mean var(B)')
#     file.close()
# =============================================================================
    
    # Reshape data-------------------------------------------------------------
    z               = z.reshape(n,m)                      # Unpack to grid
    zpredict        = zpredict.reshape(n,m)
    zpredict_SK     = zpredict_SK.reshape(n,m)
    zpredict_R      = zpredict_R.reshape(n,m)
    zpredict_SK_R   = zpredict_SK_R.reshape(n,m)
    zpredict_SK_L   = zpredict_SK_L.reshape(n,m)
    
    # For plotting MSE,R2,VAR,BIAS for different polynomials------------------
    poly_range = [1,2,3,4,5]
    
    fig12 = plt.figure(12)
    plt.plot(poly_range,R_two)
    plt.scatter(poly_range,R_two, label="OLS", marker='x', s=200)
    plt.plot(poly_range,R_two_L)
    plt.scatter(poly_range,R_two_L, label="Lasso", marker='d')
    plt.plot(poly_range,R_two_R)
    plt.scatter(poly_range,R_two_R, label="Ridge", marker='o')
    plt.xlabel('Complexity (Degree of polys)', fontsize=12)
    plt.ylabel('R2-score', fontsize=12)
    plt.legend()
    plt.title('R2-Score')
    plt.show()
    fig12.savefig('R2-score.png')
    
    fig13 = plt.figure(13)
    plt.plot(poly_range,meanerror)
    plt.scatter(poly_range,meanerror, label="OLS", marker='x', s=200)
    plt.plot(poly_range,meanerror_L)
    plt.scatter(poly_range,meanerror_L, label="Lasso", marker='d')
    plt.plot(poly_range,meanerror_R)
    plt.scatter(poly_range,meanerror_R, label="Ridge", marker='o')
    plt.xlabel('Complexity (Degree of polys)', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.legend()
    plt.title('Mean square error')
    plt.show()
    fig13.savefig('MSE.png')
    
    fig14 = plt.figure(14)
    plt.plot(poly_range,meanerror_K)
    plt.scatter(poly_range,meanerror_K, label="OLS", marker='x', s=200)
    plt.plot(poly_range,meanerror_L_K)
    plt.scatter(poly_range,meanerror_L_K, label="Lasso", marker='d')
    plt.plot(poly_range,meanerror_R_K)
    plt.scatter(poly_range,meanerror_R_K, label="Ridge", marker='o')
    plt.xlabel('Complexity (Degree of polys)', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.legend()
    plt.title('Mean square error K-fold')
    plt.show()
    fig14.savefig('MSE_Kfold.png')
    
    fig15 = plt.figure(15)
    plt.plot(poly_range,R_two_K)
    plt.scatter(poly_range,R_two_K, label="OLS", marker='x', s=200)
    plt.plot(poly_range,R_two_L_K)
    plt.scatter(poly_range,R_two_L_K, label="Lasso", marker='d')
    plt.plot(poly_range,R_two_R_K)
    plt.scatter(poly_range,R_two_R_K, label="Ridge", marker='o')
    plt.xlabel('Complexity (Degree of polys)', fontsize=12)
    plt.ylabel('R2-score', fontsize=12)
    plt.legend()
    plt.title('R2-Score K-fold')
    plt.show()
    fig15.savefig('R2-score_Kfold.png')
        
    fig16 = plt.figure(16)
    plt.plot(poly_range,A_variance)
    plt.scatter(poly_range,A_variance, label="Var OLS", marker='x', s=200)
    plt.plot(poly_range,A_variance_L)
    plt.scatter(poly_range,A_variance_L, label="Var Lasso", marker='d')
    plt.plot(poly_range,A_variance_R)
    plt.scatter(poly_range,A_variance_R, label="Var Ridge", marker='o')
    plt.plot(poly_range,A_variance_K)
    plt.scatter(poly_range,A_variance_K, label="Var K-fold OLS", marker='x', s=200)
    plt.plot(poly_range,A_variance_L_K)
    plt.scatter(poly_range,A_variance_L_K, label="Var K-fold Lasso", marker='d')
    plt.plot(poly_range,A_variance_R_K)
    plt.scatter(poly_range,A_variance_R_K, label="Var K-fold Ridge", marker='o')
    plt.xlabel('Complexity (Degree of polys)', fontsize=12)
    plt.ylabel('Variance', fontsize=12)
    plt.legend()
    plt.title('Variance')
    plt.show()
    fig16.savefig('Variance.png')
    
    fig26 = plt.figure(26)
    plt.plot(poly_range,A_bias)
    plt.scatter(poly_range,A_bias, label="Bias OLS", marker='x', s=200)
    plt.plot(poly_range,A_bias_L)
    plt.scatter(poly_range,A_bias_L, label="Bias Lasso", marker='d')
    plt.plot(poly_range,A_bias_R)
    plt.scatter(poly_range,A_bias_R, label="Bias  Ridge", marker='o')
    plt.plot(poly_range,A_bias_K)
    plt.scatter(poly_range,A_bias_K, label="Bias K-fold OLS", marker='x', s=200)
    plt.plot(poly_range,A_bias_L_K)
    plt.scatter(poly_range,A_bias_L_K, label="Bias K-fold Lasso", marker='d')
    plt.plot(poly_range,A_bias_R_K)
    plt.scatter(poly_range,A_bias_R_K, label="Bias K-fold Ridge", marker='o')
    plt.xlabel('Complexity (Degree of polys)', fontsize=12)
    plt.ylabel('Bias', fontsize=12)
    plt.legend()
    plt.title('Bias')
    plt.show()
    fig26.savefig('Bias.png')

    # For plotting graphical data------------------------------------------
    fig19 = plt.figure(19)
    ax = fig19.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, zpredict, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig19.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    plt.title('Predicted data OLS')
    plt.show()
    fig19.savefig('Predicted data OLS.png')
    
    fig20 = plt.figure(20)
    ax = fig20.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, zpredict_SK_L, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig20.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    plt.title('Predicted data LASSO')
    plt.show()
    fig20.savefig('Predicted data LASSO.png')
    
    fig21 = plt.figure(21)
    ax = fig21.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, zpredict_R, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig21.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    plt.title('Predicted data RIDGE')
    plt.show()
    fig21.savefig('Predicted data RIDGE.png')
    
    fig22 = plt.figure(22)
    ax = fig22.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig22.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    plt.title('Real data')
    plt.show()
    fig22.savefig('Real Frankie.png')
# Time ------------------------------------------------------------------
end = time.clock()
print ('CPU time:',end-start)