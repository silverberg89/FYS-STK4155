import methods
import warnings
import numpy as np
from sklearn.svm import SVC
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')

# Choices
yr              = 2012  # Year
cut             = 0.5   # Value where to decide county victory
flip            = False # If only looking at the flipped states
selection       = False # If feature importance should be active
test_selection  = False # Write test result to file
Log             = False # Logistic regression
NN              = False # Run Neural Network
plot_vars       = False # Plot votes against popolation and age over 65 for 2016
F1              = False

# Set up:
epochs = 20
batch_size = 32
eta = 0.1
lmbd = 0.001
layers = ['relu','relu','relu','sigmoid']
n_hidden_neurons = np.array([100,60,30,1])

# Score metric choice:
if F1 == True and NN == False:
    score_method = eval('f1_score')
else:
    score_method = eval('methods.accuracy')

# Load data
if flip == True:
    ''' Imports flipped data '''
    X_train,X_test,Y_train,Y_test,data,names,states,flipped = methods.swoop_data(cut)
    democrats = np.round(np.count_nonzero(Y_test)/len(Y_test),decimals=2) # % of democrats
    corr_m = methods.corr(X_train,names)
    
else:
    ''' Imports all data '''
    X_train,X_test,Y_train,Y_test,data,names,X = methods.import_data(yr,cut,0.5,binar=True)
    democrats = np.round(np.count_nonzero(Y_test)/len(Y_test),decimals=2) # % of democrats
    corr_m = methods.corr(X_train,names)
    
# For plotting variables voting percentage
if plot_vars == True:
    names_arr = np.array(names)
    var1 = str(input('Write 1:st variable: Ex: PST045214 :'))
    var1nr = np.squeeze(np.array(np.where(var1 == names_arr)))
    var2 = str(input('Write 2:nd variable: Ex: PST045214 :'))
    var2nr = np.squeeze(np.array(np.where(var2 == names_arr)))
    methods.plot_var(data,var1nr,var2nr,var1,var2)
    
# Find optimal gamma and c parameter
if Log == False: #and NN == False:
    ''' Non-linear SVM with grid search for parameters '''
    reg_vec     = np.round(np.linspace(3, 6, 10, endpoint=True))
    gamma_vec   = np.round(np.linspace(0.001, 0.02, 10, endpoint=True),decimals=7)
    test_holder = np.zeros((len(reg_vec), len(gamma_vec)), dtype=object)
    train_holder = np.zeros((len(reg_vec), len(gamma_vec)), dtype=object)
    for i, c in enumerate(reg_vec):
        for k, g in enumerate(gamma_vec):
            if NN == True:
                clf = methods.NN_keras(X_train,Y_train,epochs,eta,lmbd,n_hidden_neurons,batch_size,layers)
            else:
                clf = SVC(kernel='rbf',gamma=g,C=c,class_weight='balanced')
            clf = SVC(kernel='rbf',gamma=g,C=c,class_weight='balanced')
            clf.fit(X_train, Y_train)
            test_holder[i][k] = clf.predict(X_test)
            train_holder[i][k] = clf.predict(X_train)
    test_array = methods.visulize_para_svm(score_method,test_holder,train_holder,Y_train,Y_test,X_train,X_test,gamma_vec,reg_vec,'c','gamma')
    indicies = np.where(np.array(test_array)==np.max(np.array(test_array)))
    g = gamma_vec[indicies[0][0]]
    c = reg_vec[indicies[1][0]]
    print('C:',c,'Gamma:',g)

# Trimms down the variables by feature importance
if selection == True:
    clf     = SVC(kernel='rbf',gamma=g,C=c).fit(X_train, Y_train) # Choose SVM
    Low_v   = False                                               # Choose type of trim method
    PCA     = False
    RFE     = False
    FI      = True
    X_train_org = np.copy(X_train)
    X_test_org = np.copy(X_test)
    X_train,X_test,choosen_vars = methods.feature_analysis(Low_v,RFE,FI,PCA,clf,X_train,X_test,Y_train,pca_dim=3,threshold=0,var=10)
    if PCA == False:
        names_trimmed = []
        for i in range(len(choosen_vars)):
            if choosen_vars[i] ==True:
                names_trimmed.append(names[i])
        names = names_trimmed
    if test_selection == True:
        org_score = []
        sel_score = []
        c_vec = np.arange(1,10)
        for i,c in enumerate(c_vec):
            clf = SVC(kernel='rbf',gamma='auto',C=c,class_weight='balanced').fit(X_train_org, Y_train)
            org_score.append(score_method(Y_test,clf.predict(X_test_org)))
            clf1 = SVC(kernel='rbf',gamma='auto',C=c,class_weight='balanced').fit(X_train, Y_train)
            sel_score.append(score_method(Y_test,clf1.predict(X_test)))
        file = np.column_stack((org_score, sel_score, c_vec))
        np.savetxt('selected_variables_tes.txt', file, delimiter='\t \t',header='Orginal variables: \t Selected variables: \t C_value:',fmt='%.2f')

# Calculate all accuracy scores for all combinations of variables
''' Takes in selected design matrix and plots best combinations
    of selected features in the original data set.'''
nr_var  = int(input('How many variables?'))
combo   = list(combinations(np.arange(0,len(X_train[0])), nr_var)) # Finds all combinations
nr      = len(combo)
couter  = 0
acc_vec_test    = []                                               # Holders for accuracy score
acc_vec_train   = []
acc_vec_test_b  = []
acc_vec_train_b = []
for i in range(nr-1):                                              # Find the choosen variables
    print(i, 'of' , nr)
    X_train_itt = np.copy(X_train[:,combo[i]])
    X_test_itt  = np.copy(X_test[:,combo[i]])
    if flip == True:                                               # Need Cross validation for small data set
        kfold = int(len(X_test_itt)/30)
        for X_train_i,X_test_i,Y_train_i,Y_test_i in methods.k_fold_CV(kfold,X_train_itt,Y_train):
            couter +=1
            if Log == True:
                clf = LogisticRegression(random_state=0, solver='liblinear',multi_class='ovr')
            if NN == True:
                clf = methods.NN_keras(X_train_itt,Y_train,epochs,eta,lmbd,n_hidden_neurons,batch_size,layers)
            else:
                clf = SVC(kernel='rbf',gamma=g,C=c,class_weight='balanced')
            clf.fit(X_train_i, Y_train_i)
            acc_vec_test_b.append(score_method(Y_test_i,clf.predict(X_test_i)))
            acc_vec_train_b.append(score_method(Y_train_i,clf.predict(X_train_i)))
        acc_vec_test.append(np.sum(acc_vec_test_b)/couter)         # Average over validation sets
        acc_vec_train.append(np.sum(acc_vec_train_b)/couter)
    else:                                                          # Used for the whole data set
        couter +=1
        if Log == True:
            clf = LogisticRegression(random_state=0, solver='liblinear',multi_class='ovr')
        if NN == True:
            clf = methods.NN_keras(X_train,Y_train,epochs,eta,lmbd,n_hidden_neurons,batch_size,layers)
        else:
            clf = SVC(kernel='rbf',gamma=g,C=c,class_weight='balanced')
        clf.fit(X_train_itt, Y_train)
        acc_vec_test_b.append(score_method(Y_test,clf.predict(X_test_itt)))
        acc_vec_train_b.append(score_method(Y_train,clf.predict(X_train_itt)))
        acc_vec_test.append(np.sum(acc_vec_test_b)/couter)
        acc_vec_train.append(np.sum(acc_vec_train_b)/couter)

# Print score plot
lab = 'F1_Democratic % in testdata_'+str(democrats)+'c_'+str(c)+'_g_'+str(g)
methods.scoreplot(acc_vec_test,combo,names,lab)