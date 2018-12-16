from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import regularizers
from sklearn.metrics import f1_score

def plot_var(data,var1nr,var2nr,var1,var2):
    ''' Plot variables as democrats or republicans '''
    # Data used are from col 12:61
    var1nr = var1nr+12
    var2nr = var2nr+12
    
    fig = plt.figure()
    plt.title('Voting percentage of variable')
    plt.xlabel(var1)
    plt.ylabel('Vote percentage [%]')
    print(data.iloc[:,var1nr][0])
    plt.scatter(np.array(data.iloc[:,var1nr]),np.array(data.iloc[:,-16]),label='Democrats',c='b', alpha=0.2,marker='*')
    plt.scatter(np.array(data.iloc[:,var1nr]),np.array(data.iloc[:,-15]),label='Republicans',c='r', alpha=0.2,marker='.')
    plt.legend()
    fig.savefig('dem_vs_'+var1+'.png', dpi=300, format='png', bbox_inches='tight')
    plt.show()

    var1 = 'AGE775214'
    fig = plt.figure()
    plt.title('Voting percentage of variable')
    plt.xlabel(var2)
    plt.ylabel('Vote percentage [%]')
    print(data.iloc[:,var2nr][0])
    plt.scatter(np.array(data.iloc[:,var2nr]),np.array(data.iloc[:,-16]),label='Democrats',c='b', alpha=0.2,marker='*')
    plt.scatter(np.array(data.iloc[:,var2nr]),np.array(data.iloc[:,-15]),label='Republicans',c='r', alpha=0.2,marker='.')
    plt.legend()
    fig.savefig('dem_vs_'+var1+'.png', dpi=300, format='png', bbox_inches='tight')
    plt.show()
    return()

def linear_svm(X_train,Y_train,X_test,Y_test,names,Gamma,c):
    ''' Take in train and test data and plots the results from SVM '''
    weights = []
    clf = SVC(kernel='linear',gamma=Gamma,C=c)
    clf.fit(X_train, Y_train)
    yp_vec_test    = clf.predict(X_test)
    yp_vec_train   = clf.predict(X_train)
    acc_vec_test   = accuracy(Y_test,yp_vec_test)
    acc_vec_train  = accuracy(Y_train,yp_vec_train)
    weights.append(clf.coef_)
    weights = np.squeeze(np.array(weights))
    plot_weights('linear SVM',weights, names,90,lim=False)
    print('Accuracy score test data : ',np.round(acc_vec_test,decimals=2))
    print('Accuracy score train data: ',np.round(acc_vec_train,decimals=2))
    print('Amount of ones in test data: ',np.round(np.count_nonzero(Y_test)/len(Y_test),decimals=2))
    return()

def plot_weights(tit,values,names,deg,lim):
    ''' Take in identified variables and plots a barchart of them '''
    # Plotting
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    plt.bar(np.arange(0,len(values)), values)
    plt.title(tit)
    plt.xticks(range(0, len(values) + 1))
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xticklabels(names, rotation = deg)
    if lim == True:
        plt.ylim(min(values)-0.05, max(values)+0.05)
    plt.ylabel('Score %')
    plt.xlabel('Variables')
    fig.savefig(tit+'.png', dpi=300, format='png', bbox_inches='tight')
    plt.show()
    
def k_fold_CV(k, X, y):
    
    """
    This is a python generator. 
    Every invocation returns a new set of rows from X and y.
    The rows are never the same.  
    k is the number of rows. 
    X and y must have the same number of rows. 
    """
    
    n, m = np.shape(X)
    ind = np.arange(0, n, k)
    for i in ind:
        X_test, y_test = X[i:(i + k),:], y[i:(i + k)]
        if i == 0:
            X_train, y_train = X[i + k:, :], y[i + k:]
        elif i == (n - k):
            X_train, y_train = X[:i, :], y[:i]
        else:
            X_train = np.vstack([X[:i, :], X[i + k:, :]])
            y_train = np.hstack([y[:i], y[i + k:]])
        yield X_train, X_test, y_train, y_test 
    
def plot_features(democratics,republicans,var1,var2,var_nr1,var_nr2,yr):
    ''' Take in wanted variables and plots them against each other '''
    tit = var1+'_vs_'+var2+'_for_'+str(yr)
    fig = plt.figure()
    plt.title(tit)
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.scatter(republicans[:,var_nr1],republicans[:,var_nr2],label='Republicans',c='r', alpha=0.8,marker='.')
    plt.scatter(democratics[:,var_nr1],democratics[:,var_nr2],label='Democrats',c='b', alpha=0.3,marker='*')
    plt.legend()
    fig.savefig(tit+'.png', dpi=300, format='png', bbox_inches='tight')
    plt.show()
    return()
    
def corr(X_train,names):
    ''' Take in train design matrix and calculates the correlation matrix '''
    # Create a correlation matrix
    Corr_matrix = np.corrcoef(X_train,rowvar=False)
    high_pos_correlation = []
    for i in range(49):
        for j in range(49):
            if Corr_matrix[i,j] > 0.9 and i!=j:
                high_pos_correlation.append(names[i]+','+names[j])
    return(Corr_matrix)
    
def swoop_data(cut):
    ''' Take in data file and returns train and test data regarding flipped states '''
    data = pd.read_table('US-election-data.csv', sep = ',', header = 0, encoding = 'latin1')
    X    = np.array(data.iloc[:, 12:61])    # Design matrix
    Y12  = np.array(data.iloc[:,-4])        # Democratic % 2012
    Y16  = np.array(data.iloc[:,-16])       # Democratic % 2016
    var_names = np.array(list(data.columns.values))
    var_names = var_names[12:61]
    var_names = var_names.tolist()
    # Convert into binary data [1=victory democrats]
    Y12 = binary(Y12,cut)
    Y16 = binary(Y16,cut)
    # Normalize data
    scale = StandardScaler()
    scale.fit(X)
    X = scale.transform(X)
    # Find flipped countys
    X_flip = X[np.squeeze(np.where(Y12!=Y16)),:]
    Y_flip = Y16[np.where(Y12!=Y16)] # 0 => Flipped from dem to rep
    flipp_index = np.where(Y12!=Y16)
    states = np.array(data.iloc[:, 0])
    states = np.sort(states[np.squeeze(np.where(Y12!=Y16))])
    X_train,X_test,Y_train,Y_test = train_test_split(X_flip,Y_flip,train_size=0.5)
    return(X_train,X_test,Y_train,Y_test,data,var_names,states,flipp_index)
    
def accuracy(target,output):
    ''' Take in target and output and outputs the matric accuracy score'''
    f = np.sum(target==output) / len(target)
    return(f)
    
def binary(target,cut):
    ''' Take in y percentages and output binary data '''
    f = np.where(abs(target)>cut,1,0)
    return(f)
    
def import_data(year,cut,splitt,binar):
    ''' Take in parameters regarding the input data and outputs test and train data with nametags '''
    # Load data
    data = pd.read_table('US-election-data.csv', sep = ',', header = 0, encoding = 'latin1')
    X    = np.array(data.iloc[:, 12:61])    # Design matrix
    X_ord = np.copy(X)
    Y12  = np.array(data.iloc[:,-4])        # Democratic % 2012
    Y16  = np.array(data.iloc[:,-16])       # Democratic % 2016
    var_names = np.array(list(data.columns.values))
    var_names = var_names[12:61]
    var_names = var_names.tolist()
    # Convert into binary data [1=victory]
    if binar == True:
        Y12 = binary(Y12,cut)
        Y16 = binary(Y16,cut)
    
    # Normalize the design matrix
    scale = StandardScaler()
    scale.fit(X)
    X = scale.transform(X)
    
    # Split into training and test data
    if year == 2012:
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y12,train_size=splitt)
    if year == 2016:
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y16,train_size=splitt)
    return(X_train,X_test,Y_train,Y_test,data,var_names,X_ord)
    
def feature_analysis(low_v,rfe,fi,pca,clf,X_train,X_test,Y_train,pca_dim,threshold,var):
    ''' Take in train and test data and uses various packages for trimming the design matrix
        output is the test and train matrixes with the most valuable variables'''
    if low_v == True:
        # Remove low variance
        sel = VarianceThreshold(threshold = threshold)
        sel.fit(X_train,Y_train)
        choosen_vars = sel.get_support()
    
    if rfe == True:
        #Recursive Feature Elimination
        sel = RFE(clf, var)
        sel.fit(X_train,Y_train)
        #rfe_support_var = rfe.support_ # Selected features in the end.
        #rfe_ranking_var = rfe.ranking_ # Feature ranking, 1 = Most influence
        choosen_vars = sel.get_support()
    
    if fi == True:
        # Feature Importance
        sel = ExtraTreesClassifier()
        sel.fit(X_train,Y_train)
        #fI_importance = fI.feature_importances_
        sel = SelectFromModel(sel, prefit=True)
        choosen_vars = sel.get_support()
    
    if pca == True:
        # Principal component analysis
        sel = PCA(n_components=pca_dim)
        sel.fit(X_train,Y_train)
        #pca_expained_variance = pca.explained_variance_ratio_
        pca_components = sel.components_
        choosen_vars = pca_components
    X_train = sel.transform(X_train)
    X_test  = sel.transform(X_test)
    return(X_train,X_test,choosen_vars)
    
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
    
def NN_keras(X_train,Y_train,epochs,eta,lmbd,n_hidden_neurons,batch_size,layers):
    ''' Take in train and test data and variuous parameters, runs a neural network
        and output the fitted model as clf '''
    mid_layers = layers[1:-1]
    clf = tf.keras.Sequential()
    clf.add(tf.keras.layers.Dense(n_hidden_neurons[0], activation = str(layers[0]), input_dim = len(X_train[0]), kernel_regularizer = regularizers.l2(lmbd)))
    for i,act in enumerate(mid_layers):
        clf.add(tf.keras.layers.Dense(n_hidden_neurons[i+1], activation = str(act), kernel_regularizer = regularizers.l2(lmbd)))
    clf.add(tf.keras.layers.Dense(n_hidden_neurons[-1], activation = str(layers[-1]), kernel_regularizer = regularizers.l2(lmbd)))
    sgd = tf.keras.optimizers.SGD(lr = eta)
    clf.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])
    clf.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size, verbose = 1)
    return(clf)
    
def visulize_para_svm(score_method,test_holder,train_holder,Y_train,Y_test,X_train,X_test,gamma_vals,reg_term_vals,x_lab,y_lab):
    ''' Take in test and train holders containg results from various hyper parameters
        Outputs a heatmap of the results.
        Code similar to notes on neural networks by Morten Hjort-Jensen'''
    sns.set()
    n = len(test_holder)
    m = len(test_holder[0])
    train_accuracy = np.zeros((n, m))
    test_accuracy  = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            test_pred  = test_holder[i][j]
            train_pred = train_holder[i][j]
            train_accuracy[i][j] = score_method(Y_train,train_pred)
            test_accuracy[i][j]  = score_method(Y_test,test_pred)
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(train_accuracy, annot=True, xticklabels = reg_term_vals, yticklabels = gamma_vals, ax=ax, cmap="viridis")
    ax.set_title("Training Accuracy")
    ax.set_ylabel(y_lab)
    ax.set_xlabel(x_lab)
    plt.show()
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_accuracy, annot=True, xticklabels = reg_term_vals, yticklabels = gamma_vals, ax=ax, cmap="viridis")
    ax.set_title("Test Accuracy")
    ax.set_ylabel(y_lab)
    ax.set_xlabel(x_lab)
    plt.show()
    return(test_accuracy)
    
def scoreplot(arg,combo,names,lab):  
    ''' Take in identified variables and sends the correct data for plotting it'''   
    acc_test = np.sort(arg)
    acc_test_ind = np.argsort(arg)
    acc_test_max = acc_test[-11:-1]
    acc_test_min = acc_test[0:10]
    acc_test_ind_max = acc_test_ind[-11:-1]
    acc_test_ind_min = acc_test_ind[0:10]
    combos_max = []
    combos_min = []
    for w in range(len(acc_test_ind_max)):
        th_max = np.array(combo[acc_test_ind_max[w]])
        th_min = np.array(combo[acc_test_ind_min[w]])
        vals = []
        vals1 = []
        for j,th in enumerate(th_max):
            vals.append(names[th])
        for j1,th1 in enumerate(th_min):
            vals1.append(names[th1])
        combos_max.append(vals)
        combos_min.append(vals1)
    plot_weights('Max score '+lab,acc_test_max, combos_max,75,lim=True)
    plot_weights('Min score '+lab,acc_test_min, combos_min,75,lim=True)
    return()