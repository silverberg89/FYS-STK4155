import numpy as np
import pickle
import seaborn as sns
import regressions
import matplotlib.pyplot as plt
from Neural_Network_Act import Neural_Network_Act
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Linear_reg import Linear_reg
np.random.seed(12)

# Activation functions--------------------------------------------------
def ide_ord(x):
    ''' Linear activation function '''
    return x

def ide_der(x):
    ''' Derivative of Linear activation function '''
    return 1
    
def sig_ord(x):
    ''' Sigmoid activation function '''
    f = (1 + np.exp(-x))**(-1)
    return (f)
    
def sig_der(x):
    ''' Derivative of Sigmoid activation function '''
    f = x * (1 - x)
    return (f)

def tanh_ord(x):
    ''' Tanh activation function '''
    f = np.tanh(x)
    return (f)

def tanh_der(x):
    ''' Derivative of Tanh activation function '''
    f = 1.0 - np.tanh(x)**2
    return (f)

# Tools -------------------------------------------------------------------
    
def accuracy(target,yp):
    ''' Calculate accuracy score between target and prediction. Decide on heavyside
        where yp > 0.5 = 1, yp < 0.5 = 0 then compares the prediction class with the target class'''
    ''' Outout is accuracy score'''
    yp_bin = np.where(abs(yp)>0.5,1,0).T
    f = np.sum(target==yp_bin) / len(target)
    return (f)

def batch(x,y,batchsize):
    ''' Divides input data into batches for logistic regression '''
    ''' Output is two lists of batches'''
    y_batch = []
    x_batch = []
    sectionpoints = np.arange(0,len(x),batchsize)
    for i in sectionpoints:
        y_batch.append(y[i:i + batchsize])
        x_batch.append(x[i:i + batchsize])
    return(x_batch,y_batch,len(sectionpoints))
    
def mse(y, yp):
    ''' Mean Square Error'''
    ''' Returns MSE'''
    n = len(y)
    mse = sum((y-yp)**2)*(1 / n)
    return (mse)

def r2(y, yp):
    '''R2-Score'''
    ''' Returns R2 Score'''
    r2 = 1 - (sum((y-yp)**2) / sum((y - sum(y) / float(y.size))**2))
    return (r2)

def r2_boot(y, yp):
    '''R2-Score for bootstrap'''
    ''' Returns R2 Score'''
    print(yp.shape)
    m = len(yp[0])
    r2_arr = np.zeros(m)
    for i in range(m):
        r2_arr[i] = (1 - (sum((y-yp[:,i])**2) / sum((y - sum(y) / float(y.size))**2)))
    rs_ret = np.mean(r2_arr)
    return (rs_ret)

def bias2(y,yp):
    ''' Bias^2'''
    '''returns BIAS^2'''
    y = y[:,np.newaxis]
    yp = yp[:,np.newaxis]
    B = np.mean( (y - np.mean(yp, axis=1, keepdims=True))**2 )
    return(B)

def variance(yp):
    '''Calculates the variance'''
    '''Returns variance'''
    yp = yp[:,np.newaxis]
    G = np.mean( np.var(yp, axis=1, keepdims=True) )
    return(G)
    
def error(y,yp):
    ''' Computes the MSE'''
    ''' Returns mse'''
    y = y[:,np.newaxis]
    yp = yp[:,np.newaxis]
    E = np.mean( np.mean((y - yp)**2, axis=1, keepdims=True) )
    return(E)
    
def boot(x_train,y_train,x_test,y_test,n_boostraps,pen_r,pen_l):
    ''' Preform a bootstrap on linear regression.
        Code similar to notes on piazza by Bendik Samseth'''
    '''Output is statistics on the input data'''
    yp_ols = np.empty((y_test.shape[0], n_boostraps))
    yp_rid = np.empty((y_test.shape[0], n_boostraps))
    yp_las = np.empty((y_test.shape[0], n_boostraps))
    for i in range(n_boostraps):
        x_boot, y_boot = resample(x_train, y_train)
        regit = Linear_reg(y_boot,x_test,x_boot,y_test)
        regit.ols()
        yp_ols[:, i] = regit.yp_ols
        regit.ridge(pen_r)
        yp_rid[:, i] = regit.yp_rid
        regit.lasso(pen_l)
        yp_las[:, i] = regit.yp_las
        print('Bootstrap: ',i,' of ',n_boostraps)
    
    stats = np.zeros((3,4))         # (Ols, Ridge, Lasso) x (Var,Bias,MSE,R2)
    yp_vec = [yp_ols,yp_rid,yp_las]
    for k, yp in enumerate(yp_vec): # Run R2 score
        stats[k,3] = r2_boot(y_test, yp)
        
    y_test  = y_test[:,np.newaxis] # For taking elements in correct axis
    for k, yp in enumerate(yp_vec):# Run MSE,Bias,Var
        mserror = np.mean( np.mean((y_test - yp)**2, axis=1, keepdims=True) )
        biastwo = np.mean( (y_test - np.mean(yp, axis=1, keepdims=True))**2 )
        var     = np.mean( np.var(yp, axis=1, keepdims=True) )
        stats[k,0:3]  = np.asarray([var,biastwo,mserror])
    return(stats)
    
def boot_NN(X_train,Y_train,Y_test,X_test,epochs,batch_size,hidden_neurons,right_lay_dim,learn_r,reg_term,costfunc,act_I,act_H,act_O,n_boostraps):
    ''' Preform a bootstrap on neural networks regarding regression cases.
        Output is a matrix holding:  [variance, bias2, mse and r2 score]'''
    yp_vec = np.empty((Y_test.shape[0], n_boostraps))
    for i in range(n_boostraps):
        x_boot, y_boot = resample(X_train, Y_train)
        Neu_Ising                       = Neural_Network_Act(hidden_neurons,right_lay_dim,learn_r,reg_term,costfunc,act_I,act_H,act_O)
        loss_epoch_history              = Neu_Ising.mini_batch_train(x_boot,y_boot,epochs,batch_size)
        yp       ,w_neu_te,b_neu_te     = Neu_Ising.predict(X_test)
        yp_vec[:,i] = np.squeeze(yp)
        
    stats_boot = np.zeros((1,4))
    stats_boot[0,3] = r2_boot(Y_test, yp)
    Y_test  = Y_test[:,np.newaxis]
    mserror = np.mean( np.mean((Y_test - yp)**2, axis=1, keepdims=True) )
    biastwo = np.mean( (Y_test - np.mean(yp, axis=1, keepdims=True))**2 )
    var     = np.mean( np.var(yp, axis=1, keepdims=True) )
    stats_boot[0,0:3]  = np.asarray([var,biastwo,mserror])
    return(stats_boot)

# Ising data and plots-------------------------------------------------------
    
def ising_energies(states,L):
    """ This function calculates the energies of the states in the nn Ising Hamiltonian
        code similar to metha """
    J=np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    # Compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)
    return E
    
def isning_data(splitt): 
    ''' Reads and construct data from ising model, code similar to metha'''
    data = pickle.load(open('IsingData/'+'Ising2DFM_reSample_L40_T=All.pkl','rb'))
    data = np.unpackbits(data).reshape(-1, 1600)
    data=data.astype('int')
    labels = pickle.load(open('IsingData/'+'Ising2DFM_reSample_L40_T=All_labels.pkl','rb'))
    
    # Divide data into ordered, critical and disordered
    train_to_test_ratio = splitt
    X_ordered=data[:70000,:]
    Y_ordered=labels[:70000]
    X_O_train,X_O_test,Y_O_train,Y_O_test=train_test_split(X_ordered,Y_ordered,train_size=train_to_test_ratio)
    X_critical=data[70000:100000,:]
    Y_critical=labels[70000:100000]
    X_C_train,X_C_test,Y_C_train,Y_C_test=train_test_split(X_critical,Y_critical,train_size=train_to_test_ratio)
    X_disordered=data[100000:,:]
    Y_disordered=labels[100000:]
    X_D_train,X_D_test,Y_D_train,Y_D_test=train_test_split(X_disordered,Y_disordered,train_size=train_to_test_ratio)
    X = np.concatenate((X_ordered,X_disordered))
    Y = np.concatenate((Y_ordered,Y_disordered))
    return(X,Y)
    
def ising_data1D(n,L):
    ''' Construct data from for ising model, code similar to metha'''
    # Data creation
    states       = np.random.choice([-1, 1], size=(n,L))
    energies     = ising_energies(states,L)   # Ising energies vector [y]
    # Reshape Ising states into RL samples: S_iS_j --> X_p
    states=np.einsum('...i,...j->...ij', states, states)
    shape=states.shape
    states=states.reshape((shape[0],shape[1]*shape[2]))
    # Create set
    Data=[states,energies]
    return(Data)
    
def identify_hyper_log(x_train,y_train,x_test,y_test,reg_terms,l_rate,epochs,batchsize):
    ''' Loop over different regularization values and compare against accuracy
        score with a plot at the end. Code from Metha'''
    ''' Plots results'''
    sk_test_accuracy_SGD=np.zeros(reg_terms.shape,np.float64)
    code_test_accuracy_SGD=np.zeros(reg_terms.shape,np.float64)
    sk_train_accuracy_SGD=np.zeros(reg_terms.shape,np.float64)
    code_train_accuracy_SGD=np.zeros(reg_terms.shape,np.float64)
    for i,reg_term in enumerate(reg_terms):
        # Regression using SGD code
        W,B                     = regressions.logistic(x_train,y_train,batchsize,epochs,l_rate,reg_term)
        yp_te                   = sig_ord(np.dot(x_test,W)+B)
        yp_tr                   = sig_ord(np.dot(x_train,W)+B)
        acc_trsk,acc_tesk       = regressions.logistic_sk(x_test,y_test,x_train,y_train,epochs,l_rate,reg_term)
        # Accuracy score holders
        sk_test_accuracy_SGD[i]=acc_tesk
        code_test_accuracy_SGD[i]=accuracy(y_test,yp_te)
        sk_train_accuracy_SGD[i]=acc_trsk
        code_train_accuracy_SGD[i]=accuracy(y_train,yp_tr)
        print('SGD   : %0.4f, %0.4f,' %(sk_test_accuracy_SGD[i],code_test_accuracy_SGD[i]))
        print('SGD   : %0.4f, %0.4f,' %(sk_train_accuracy_SGD[i],code_train_accuracy_SGD[i]))
    # Plot
    plt.semilogx(reg_terms,sk_test_accuracy_SGD,'*-b',label='Sklearn test', linewidth=1.5)
    plt.semilogx(reg_terms,code_test_accuracy_SGD,'*--g',label='Code test', linewidth=1.5)
    plt.xlabel('$\\lambda$ (Reg_term)')
    plt.ylabel('$\\mathrm{accuracy}$')
    plt.grid()
    plt.legend()
    plt.show()
    
def identify_epochs(x_train,y_train,x_test,y_test,reg_term,l_rate,epoch_vec,batchsize):
    ''' Loop over different epoch settings and compare against accuracy
        score with a plot at the end. Code from Metha'''
    ''' Plots results'''
    train_accuracy_SGD=np.zeros(epoch_vec.shape,np.float64)
    test_accuracy_SGD=np.zeros(epoch_vec.shape,np.float64)
    for i,epochs in enumerate(epoch_vec):
        # Regression using SGD code
        W                = regressions.logistic(x_train,y_train,batchsize,epochs,l_rate,reg_term)
        yp               = sig_ord(np.dot(x_test,W))
        # Accuracy score holders
        train_accuracy_SGD[i]=accuracy(y_train,yp)
        test_accuracy_SGD[i]=accuracy(y_test,yp)
        print('SGD   : %0.4f, %0.4f,' %(train_accuracy_SGD[i],test_accuracy_SGD[i]))
    # Plot
    plt.semilogx(epoch_vec,train_accuracy_SGD,'*-g',label='SGD train')
    plt.semilogx(epoch_vec,test_accuracy_SGD,'*--g',label='SGD test')
    plt.xlabel('$\\lambda$ (Reg_term)')
    plt.ylabel('$\\mathrm{accuracy}$')
    plt.grid()
    plt.legend()
    plt.show()
    
def identify_hyperpara(plot,hidden_neurons,right_lay_dim,learn_r,reg_term,costfunc,X_test,Y_test,X_train,Y_train,epochs,batch_size,learn_r_vals,reg_term_vals,act_I,act_H,act_O):
    ''' Run the neural network for different values of reg_term (lambda) and
        learning rate (eta). Collect weights and biases and print accuracy score.
        Code from notes on neural networks by Morten Hjort-Jensen'''
    ''' Plots results'''
    test_holder  = np.zeros((len(learn_r_vals), len(reg_term_vals)), dtype=object)
    train_holder = np.zeros((len(learn_r_vals), len(reg_term_vals)), dtype=object)
    for i, learn_r in enumerate(learn_r_vals):
        '''# Generally bad method for changing neurons,epochs and reg_terms, it works.. not a priority to change.
            # Ensure that max(neurons) > max(epochs).. :)'''
        counter = 0
        if type(reg_term_vals) is list:
            y_lab = 'Learning rate'
            x_lab = 'Neurons'
            for j, hidden_neurons in enumerate(reg_term_vals):
                counter += 1
                Neu_Ising                       = Neural_Network_Act(hidden_neurons,right_lay_dim,learn_r,reg_term,costfunc,act_I,act_H,act_O)
                cost_epoch_history              = Neu_Ising.mini_batch_train(X_train,Y_train,epochs,batch_size)
                yp_test,w_neu_te,b_neu_te       = Neu_Ising.predict(X_test)
                yp_train,w_neu_tr,b_neu_tr      = Neu_Ising.predict(X_train)
                test_holder[i][j]               = yp_test
                train_holder[i][j]              = yp_train
                print("Learning rate  = ", learn_r)
                print("Neurons = ", reg_term)
                print("Accuracy score on test set: ", accuracy(Y_test,test_holder[i][j]))
                print("Accuracy score on train set: ", accuracy(Y_train,train_holder[i][j]))
                print(counter, 'of 196')
        elif min(reg_term_vals) > 2 and max(reg_term_vals) < 400:
            y_lab = 'Learning rate'
            x_lab = 'Epochs'
            for j, epochs in enumerate(reg_term_vals):
                counter += 1
                Neu_Ising                       = Neural_Network_Act(hidden_neurons,right_lay_dim,learn_r,reg_term,costfunc,act_I,act_H,act_O)
                cost_epoch_history              = Neu_Ising.mini_batch_train(X_train,Y_train,epochs,batch_size)
                yp_test,w_neu_te,b_neu_te       = Neu_Ising.predict(X_test)
                yp_train,w_neu_tr,b_neu_tr      = Neu_Ising.predict(X_train)
                test_holder[i][j]               = yp_test
                train_holder[i][j]              = yp_train
                print("Learning rate  = ", learn_r)
                print("Epochs = ", reg_term)
                print("Accuracy score on test set: ", accuracy(Y_test,test_holder[i][j]))
                print("Accuracy score on train set: ", accuracy(Y_train,train_holder[i][j]))
                print(counter, 'of 196')
        elif min(reg_term_vals) > 7 and max(reg_term_vals) < 2500:
            y_lab = 'Learning rate'
            x_lab = 'Batch Size'
            for j, batch_size in enumerate(reg_term_vals):
                counter += 1
                Neu_Ising                       = Neural_Network_Act(hidden_neurons,right_lay_dim,learn_r,reg_term,costfunc,act_I,act_H,act_O)
                cost_epoch_history              = Neu_Ising.mini_batch_train(X_train,Y_train,epochs,batch_size)
                yp_test,w_neu_te,b_neu_te       = Neu_Ising.predict(X_test)
                yp_train,w_neu_tr,b_neu_tr      = Neu_Ising.predict(X_train)
                test_holder[i][j]               = yp_test
                train_holder[i][j]              = yp_train
                print("Learning rate  = ", learn_r)
                print("Batch Size = ", batch_size)
                print("Accuracy score on test set: ", accuracy(Y_test,test_holder[i][j]))
                print("Accuracy score on train set: ", accuracy(Y_train,train_holder[i][j]))
                print(counter, 'of 196')
        else:
            y_lab = 'Learning rate'
            x_lab = 'Regularization term value'
            for j, reg_term in enumerate(reg_term_vals):
                counter += 1
                Neu_Ising                       = Neural_Network_Act(hidden_neurons,right_lay_dim,learn_r,reg_term,costfunc,act_I,act_H,act_O)
                cost_epoch_history              = Neu_Ising.mini_batch_train(X_train,Y_train,epochs,batch_size)
                yp_test,w_neu_te,b_neu_te       = Neu_Ising.predict(X_test)
                yp_train,w_neu_tr,b_neu_tr      = Neu_Ising.predict(X_train)
                test_holder[i][j]               = yp_test
                train_holder[i][j]              = yp_train
                print("Learning rate  = ", learn_r)
                print("Reg_term = ", reg_term)
                print("Accuracy score on test set: ", accuracy(Y_test,test_holder[i][j]))
                print("Accuracy score on train set: ", accuracy(Y_train,train_holder[i][j]))
                print(counter, 'of 196')
    if plot == 1:
        print('Prepare plotting of hyperparameters')
        visulize_hyperpara(test_holder,train_holder,Y_train,Y_test,X_train,X_test,learn_r_vals,reg_term_vals,x_lab,y_lab)
    
def visulize_hyperpara(test_holder,train_holder,Y_train,Y_test,X_train,X_test,learn_r_vals,reg_term_vals,x_lab,y_lab):
    ''' Take in matrices of weights and biases for different values of
        reg_term (lambda) and learning rate (eta). Produces a heatmap.
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
            train_accuracy[i][j] = accuracy(Y_train,train_pred)
            test_accuracy[i][j]  = accuracy(Y_test,test_pred)
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(train_accuracy, annot=True, xticklabels = reg_term_vals, yticklabels = learn_r_vals, ax=ax, cmap="viridis")
    ax.set_title("Training Accuracy")
    ax.set_ylabel(y_lab)
    ax.set_xlabel(x_lab)
    plt.show()
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_accuracy, annot=True, xticklabels = reg_term_vals, yticklabels = learn_r_vals, ax=ax, cmap="viridis")
    ax.set_title("Test Accuracy")
    ax.set_ylabel(y_lab)
    ax.set_xlabel(x_lab)
    plt.show()
    return()
    
def plot_J(J_ols,Jr,Jl,L,penalty):
    ''' PLots OLS, RIDGE and LASSO coupling constants for different penalties
        Code by Mehta et al..'''
    J_leastsq=np.array(J_ols).reshape((L,L))
    J_ridge=np.array(Jr).reshape((L,L))
    J_lasso=np.array(Jl).reshape((L,L))
    cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')
    fig, axarr = plt.subplots(nrows=1, ncols=3)
    axarr[0].imshow(J_leastsq,**cmap_args)
    axarr[0].set_title('$\\mathrm{OLS}$',fontsize=16)
    axarr[0].tick_params(labelsize=16)
    axarr[1].imshow(J_ridge,**cmap_args)
    axarr[1].set_title('$\\mathrm{Ridge},\ \\lambda=%.4f$' %(penalty),fontsize=16)
    axarr[1].tick_params(labelsize=16)
    im=axarr[2].imshow(J_lasso,**cmap_args)
    axarr[2].set_title('$\\mathrm{LASSO},\ \\lambda=%.4f$' %(penalty),fontsize=16)
    axarr[2].tick_params(labelsize=16)
    divider = make_axes_locatable(axarr[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar=fig.colorbar(im, cax=cax)
    cbar.ax.set_yticklabels(np.arange(-1.0, 1.0+0.25, 0.25),fontsize=14)
    cbar.set_label('$J_{i,j}$',labelpad=-40, y=1.12,fontsize=16,rotation=0)
    fig.subplots_adjust(right=2.0)
    plt.show()