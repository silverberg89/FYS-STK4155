import time
import warnings
import logmethods
import regressions
import numpy as np
from Linear_reg import Linear_reg
from Neural_Network_Act import Neural_Network_Act
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
np.random.seed(12)

print('1: 1D, Linear reg, Coupling constant [b]')
print('2: 2D, Logistic reg, Phase determination [c]')
print('3: 1D, Multilayer net, Weights & Biases [d]')
print('4: 2D, Multilayer net, Phase determination [e]')
task = input('Choose task: ')

if int(task) == 4:
    ''' Identifies phase change on a 40 by 40 lattice with spins in 2D
        This is achived by a neural network with a mini-batch stochastic
        gradient solver and measured by accuracy score '''
    T0 = time.clock()
    # Load data:
    splitt = 0.5
    X,Y = logmethods.isning_data(splitt)
    # Create training and test sets:
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=splitt)
    # Data parameters:
    nr_inputs      = len(X_train)   # Training set size
    left_lay_dim   = len(X_train[0])# Input layer dimensionality
    right_lay_dim  = 1              # Output layer dimensionality
    hidden_neurons = [50]           # Nr of neurons in respective layer
    reg_term       = 0.01          # Regularizaion term
    learn_r        = 0.001          # Learning rate
    batch_size     = 64
    epochs         = 5
    # Neural network choices
    act_I          = 'sig'   # Among [sig, ide, tanh]
    act_H          = 'sig'   # Among [sig, ide, tanh]
    act_O          = 'sig'   # Among [sig, ide, tanh]
    costfunc       = 1       # 1 = cross_entropy (Classification), 0 = quadratic (Regression)
    test_para      = 1       # 1 if you want to test different parameters
    # Run the calculation:
    Neu_Ising                       = Neural_Network_Act(hidden_neurons,right_lay_dim,learn_r,reg_term,costfunc,act_I,act_H,act_O)
    cost_epoch_history              = Neu_Ising.mini_batch_train(X_train,Y_train,epochs,batch_size)
    yp_test,w_neu_te,b_neu_te       = Neu_Ising.predict(X_test)
    yp_train,w_neu_tr,b_neu_tr      = Neu_Ising.predict(X_train)
    # Calculate accuracy score:
    Acc_Score_te                    = logmethods.accuracy(Y_test,yp_test)
    Acc_Score_tr                    = logmethods.accuracy(Y_train,yp_train)
    print('Acc_Score Test: ',Acc_Score_te)
    print('Acc_Score Train: ',Acc_Score_tr)
    
    # Identify hyperparameters
    if test_para == 1:
        plot                = 1 # 1 for plotting heatmap
        learn_r_vals        = np.logspace(-5, 1, 7)
        reg_term_vals       = np.logspace(-5, 1, 7)
        neuron_vals         = [[100,80],[80,40],[40,20],[40,40,40],[20,40],[40,80],[80,100]]
        epoch_vals          = np.array([3,10,20,30,50,70,100])
        batch_vals          = np.array([8,16,32,64,100,200,400])
        
        logmethods.identify_hyperpara(plot,hidden_neurons,right_lay_dim,learn_r,reg_term,costfunc,X_test,Y_test,X_train,Y_train,epochs,batch_size,learn_r_vals,reg_term_vals,act_I,act_H,act_O)
        logmethods.identify_hyperpara(plot,hidden_neurons,right_lay_dim,learn_r,reg_term,costfunc,X_test,Y_test,X_train,Y_train,epochs,batch_size,learn_r_vals,neuron_vals,act_I,act_H,act_O)
        logmethods.identify_hyperpara(plot,hidden_neurons,right_lay_dim,learn_r,reg_term,costfunc,X_test,Y_test,X_train,Y_train,epochs,batch_size,learn_r_vals,epoch_vals,act_I,act_H,act_O)
        logmethods.identify_hyperpara(plot,hidden_neurons,right_lay_dim,learn_r,reg_term,costfunc,X_test,Y_test,X_train,Y_train,epochs,batch_size,learn_r_vals,batch_vals,act_I,act_H,act_O)
        
    print('Computational Time: ',time.clock()-T0)

if int(task) == 3:
    ''' Uses a multilayer neural network with backpropagation to find the
        optimal weights and biases. The result is measured by R2 score
        and MSE '''
    T0 = time.clock()
    # Parameters
    L              = 40      # System size
    n              = 10000
    right_lay_dim  = 1       # Output layer dimensionality
    hidden_neurons = [40,10] # Nr of neurons in respective layer
    reg_term       = 0.01    # Regularizaion term
    learn_r        = 0.0002  # Learning rate
    batch_size     = 32
    epochs         = 20
    # Neural network choices
    act_I          = 'sig'   # Among [sig, ide, tanh]
    act_H          = 'ide'   # Among [sig, ide, tanh]
    act_O          = 'ide'   # Among [sig, ide, tanh]
    costfunc       = 0       # 1 = cross_entropy (Classification), 0 = quadratic (Regression)
    test_para      = 0       # 1 if you want to test different parameters
    boot_run       = 0
    n_boostraps    = 50
    # Draws data
    Data = logmethods.ising_data1D(n,L)
    # Split data
    X_train,X_test,Y_train,Y_test=train_test_split(Data[0],Data[1],train_size=0.75)
    # Run the calculation:
    Neu_Ising                       = Neural_Network_Act(hidden_neurons,right_lay_dim,learn_r,reg_term,costfunc,act_I,act_H,act_O)
    loss_epoch_history              = Neu_Ising.mini_batch_train(X_train,Y_train,epochs,batch_size)
    yp_neu_te,w_neu_te,b_neu_te     = Neu_Ising.predict(X_test)
    yp_neu_tr,w_neu_tr,b_neu_tr     = Neu_Ising.predict(X_train)
    # Statistics
    r2_neu_te       = logmethods.r2_boot(Y_test,yp_neu_te)
    mse_neu_te      = logmethods.mse(Y_test,np.squeeze(yp_neu_te))
    bias2_neu_te    = logmethods.bias2(Y_test,np.squeeze(yp_neu_te))
    variance_neu_te = logmethods.variance(np.squeeze(yp_neu_te))
    r2_neu_tr       = logmethods.r2_boot(Y_train,yp_neu_tr)
    mse_neu_tr      = logmethods.mse(Y_train,np.squeeze(yp_neu_tr))
    bias2_neu_tr    = logmethods.bias2(Y_train,np.squeeze(yp_neu_tr))
    variance_neu_tr = logmethods.variance(np.squeeze(yp_neu_tr))
    # Bootstrap
    if boot_run == 1:
        stats_boot = logmethods.boot_NN(X_train,Y_train,Y_test,X_test,epochs,batch_size,hidden_neurons,right_lay_dim,learn_r,reg_term,costfunc,act_I,act_H,act_O,n_boostraps)
    
    # Finding optimal parameters
    if test_para == 1:
        #lr_vec = [0.0002,0.0001,0.00001]
        hid    = [[30,20],[40,20],[40,10]]
        test_arr  = np.zeros((len(hid),4))
        test_arr_boot = []
        for i,hidden_neurons in enumerate(hid):
            # Run the calculation:
            Neu_Ising                       = Neural_Network_Act(hidden_neurons,right_lay_dim,learn_r,reg_term,costfunc,act_I,act_H,act_O)
            loss_epoch_history              = Neu_Ising.mini_batch_train(X_train,Y_train,epochs,batch_size)
            yp_neu_te,w_neu_te,b_neu_te     = Neu_Ising.predict(X_test)
            # Statistics test
            test_arr[i,0] = logmethods.r2_boot(Y_test,yp_neu_te)
            test_arr[i,1] = logmethods.mse(Y_test,np.squeeze(yp_neu_te))
            test_arr[i,2] = logmethods.bias2(Y_test,np.squeeze(yp_neu_te))
            test_arr[i,3] = logmethods.variance(np.squeeze(yp_neu_te))
            if boot_run == 1:
                stats = logmethods.boot_NN(X_train,Y_train,Y_test,X_test,epochs,batch_size,hidden_neurons,right_lay_dim,learn_r,reg_term,costfunc,act_I,act_H,act_O,n_boostraps)
                test_arr_boot.append(stats)
    
    print('Computational Time: ',time.clock()-T0)
if int(task) == 2: 
    ''' Identifies phase on a 40 by 40 lattice with spins in 2D
        This is achived by logistic regression with a mini-batch stochastic
        gradient solver and measured by accuracy score '''
    T0 = time.clock()
    # Parameters
    batchsize   = 16                        # Size of batches in SGD
    epochs      = 8                        # Number of epochs in SGD
    l_rate      = 0.0001                    # Learning rate in SGD
    reg_term    = 0.01
    split_size  = 0.5                       # Test to Train ratio
    test_reg    = 1                         # 0 = No reg_term test
    test_epoch  = 0                         # 0 = no epoch test
    # Create data, Y[:70000] = Ordered = 1, Y[100000:] = Disordered = 0
    X,Y = logmethods.isning_data(split_size)
    # Split data into traning and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=split_size)
    # Regression using SGD code
    W,B                 = regressions.logistic(X_train,Y_train,batchsize,epochs,l_rate,reg_term)
    yp                  = logmethods.sig_ord(np.dot(X_test,W) + B)
    yp_t                = logmethods.sig_ord(np.dot(X_train,W) + B)
    acc_score           = logmethods.accuracy(Y_test,yp)
    acc_score_te        = logmethods.accuracy(Y_train,yp_t)
    # Regression using SGD Sklearn
    acc_sk_tr,acc_sk_te = regressions.logistic_sk(X_test,Y_test,X_train,Y_train,epochs,l_rate,reg_term)
    if test_reg == 1:
        # Test different reg_terms against accuracy
        reg_terms = np.logspace(-5,5,11)
        logmethods.identify_hyper_log(X_train,Y_train,X_test,Y_test,reg_terms,l_rate,epochs,batchsize)
    if test_epoch == 1:
        # Test different epochs against accuracy
        epoch_vec = np.arange(5,300,50)
        logmethods.identify_epochs(X_train,Y_train,X_test,Y_test,reg_term,l_rate,epoch_vec,batchsize)
    print((time.clock() - T0))
    
if int(task) == 1:
    ''' Identifies the coupling constant (J) of the model by linear regression.
        It includes resampling by bootstrap and measure bias, variance, R2 and MSE.'''
    T0 = time.clock()
    # Parameters
    np.random.seed(12)      # Choose random state
    L             = 40      # System size
    n             = 10000
    bootstraps    = 2
    pen_r         = 1
    pen_l         = 0.0001
    boot_         = 0 # Choose if bootstaping and plotting should run, 0 = no
    plot_         = 0
    # Draws data
    Data = logmethods.ising_data1D(n,L)
    # Split data
    X_train,X_test,Y_train,Y_test=train_test_split(Data[0],Data[1],train_size=0.25)
    # Linear regression start here
    reg = Linear_reg(Y_train,X_test,X_train,Y_test)
    # OLS
    reg.ols()
    yp_ols       = reg.yp_ols
    J_ols        = reg.J_ols
    r2_ols       = reg.r2_ols
    mse_ols      = reg.mse_ols
    bias2_ols    = reg.bias2_ols
    variance_ols = reg.variance_ols
    print('R2 OLS: ',"%.5f" % r2_ols,',','MSE OLS: ',"%.5f" % mse_ols)
    print('Bias OLS: ',"%.5f" % bias2_ols,',','Variance OLS: ',"%.5f" % variance_ols)
    # Ridge and Lasso for different penalties and final plot J
    penalties    = np.logspace(-4, 5, 10)
    g            = len(penalties)
    J_rid        = []
    J_las        = []
    yp_rid       = []
    yp_las       = []
    r2_rid       = np.zeros(g)
    mse_rid      = np.zeros(g)
    bias2_rid    = np.zeros(g)
    variance_rid = np.zeros(g)
    r2_las       = np.zeros(g)
    mse_las      = np.zeros(g)
    bias2_las    = np.zeros(g)
    variance_las = np.zeros(g)
    for i,penalty in enumerate(penalties):
        reg.ridge(penalty)
        yp_ridx         = reg.yp_rid
        yp_rid.append(yp_ridx)
        J_ridx          = reg.J_rid
        J_rid.append(J_ridx)
        r2_rid[i]       = reg.r2_rid
        mse_rid[i]      = reg.mse_rid
        bias2_rid[i]    = reg.bias2_rid
        variance_rid[i] = reg.variance_rid
        
        reg.lasso(penalty)
        yp_lasx         = reg.yp_las
        yp_las.append(yp_lasx)
        J_lasx          = reg.J_las
        J_las.append(J_lasx)
        r2_las[i]       = reg.r2_las
        mse_las[i]      = reg.mse_las
        bias2_las[i]    = reg.bias2_las
        variance_las[i] = reg.variance_las
        
        if plot_ == 1:
            # The coupling constants are along the diagonals
            logmethods.plot_J(J_ols,J_ridx,J_lasx,L,penalty)
    del (J_lasx,J_ridx,yp_lasx,yp_ridx)
    # Start bootstrapping with choosen penalties in parameters
    if boot_ == 1:
        # For 'stats_re' read rows as: (Ols, Ridge, Lasso) and columns: (Var,Bias,MSE,R2)
        stats_re = logmethods.boot(X_train,Y_train,X_test,Y_test,bootstraps,pen_r,pen_l)

    print(time.clock()-T0)