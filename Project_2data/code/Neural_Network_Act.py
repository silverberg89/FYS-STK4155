import numpy as np
import logmethods
np.random.seed(12)

class Neural_Network_Act():
    ''' Constructs a neural network and preform either classification
        or predictions for a regression case.'''
    def __init__(
        self,
        hidden_neurons,
        right_lay_dim,
        learn_r,
        reg_term,
        costfunc,
        act_I,
        act_H,
        act_O,):
        self.hidden_layers  = len(hidden_neurons) # Nr of hidden layers
        self.hidden_neurons = hidden_neurons      # List of hidden neurons
        self.right_lay_dim  = right_lay_dim       # NR of output neurons
        self.learn_r        = learn_r             # Learning rate
        self.reg_term       = reg_term            # Penalty parameter
        self.costfunc       = costfunc            # Choice of cost function
        self.act_I          = eval('logmethods.'+act_I+'_ord') # Activation functions
        self.act_H          = eval('logmethods.'+act_H+'_ord')
        self.act_O          = eval('logmethods.'+act_O+'_ord')
        self.act_D          = eval('logmethods.'+act_O+'_der')

    def arrays(self):
        ''' Create arrays for the network'''
        g = np.zeros(self.hidden_layers+1, dtype = object)
        return(g)

    def biases_and_weights(self,x):
        ''' Sets up arrays for the neural network and fills weights
            and bias vectors with normaly distrubuted randomized elements.'''
        self.z           = self.arrays()
        self.active_z    = self.arrays()
        self.error       = self.arrays()
        self.weights     = self.arrays()
        self.bias        = self.arrays()
        self.examples    = len(x[0])
        
        # Fill weights and biases with initial random elements and normalize
        for i in range(1,self.hidden_layers):
           self. weights[i] = np.random.randn(self.hidden_neurons[i-1],self.hidden_neurons[i])
           self.bias[i]     = np.random.randn(self.hidden_neurons[i]) + 0.01 # Add 0.01 for ensuring first back_prop cycle
        self.weights[0]   = np.random.randn(self.examples,self.hidden_neurons[0])
        self. weights[-1] = np.random.randn(self.hidden_neurons[-1],self.right_lay_dim)
        self.bias[0]      = np.random.randn(self.hidden_neurons[0]) + 0.01
        self. bias[-1]    = np.random.randn(self.right_lay_dim) + 0.01

    def feed_forward(self):
        ''' Calculate forward in the net with up-to-date weights and biases'''
        # Activate neurons for input layer
        self.z[0]        = np.dot(self.batch_x,self.weights[0]) + self.bias[0]
        self.active_z[0] = self.act_I(self.z[0])
        # Activate neurons for hidden layers
        for i in range(1,self.hidden_layers):
            self.z[i]        = np.dot(self.active_z[i-1],self.weights[i]) + self.bias[i]
            self.active_z[i] = self.act_H(self.z[i])
        # Activate neurons for output layer
        self.z[-1]        = np.dot(self.active_z[-2],self.weights[-1]) + self.bias[-1]
        self.active_z[-1] = self.act_O(self.z[-1])
        
    def back_prop(self):
        ''' Perform the backpropagation algorithm with choice of either
            the Quadratic or the Cross entropy cost-function
            "Side: right(r),left(l)" is with respect to the processed layer.'''
        # Last layer error (Output)
        if self.costfunc == 1:
            # Using gradient of cross entropy costfunction for classification
            self.error[-1] = (self.active_z[-1] - self.batch_y[np.newaxis,:].T)
            
        else:
            # Using gradient of Quadratic costfunction for regression
            self.error[-1] = (self.active_z[-1] - self.batch_y[np.newaxis,:].T) * self.act_D(self.active_z[-1])
            
        # Error in backward layers (Input <-- Hidden)
        for i in range(self.hidden_layers-1,-1,-1):
            self.error[i]  = np.dot(self.error[i+1],self.weights[i+1].T) * self.act_D(self.active_z[i])
            
        # Calculate partial derivitives of costfunction and update left side
        delta_weight_left   = np.dot(self.batch_x.T,self.error[0])
        delta_bias_left     = np.sum(self.error[0],axis=0)
        
        # Regulization and update of input connections for left side
        delta_weight_left   += self.reg_term * self.weights[0]
        delta_bias_left     += self.reg_term * self.bias[0] 
        self.weights[0]     -= self.learn_r * delta_weight_left
        self.bias[0]        -= self.learn_r * delta_bias_left
        
        # Calculate partial derivitives of costfunction and update right side
        for i in range(1,self.hidden_layers+1):
            delta_weight_right  = np.dot(self.active_z[i-1].T,self.error[i])
            delta_bias_right    = np.sum(self.error[i],axis=0)
            
            # Regulization and update of right side for right side
            delta_weight_right  += self.reg_term * self.weights[i]
            delta_bias_right    += self.reg_term * self.bias[i] 
            self.weights[i]     -= self.learn_r * delta_weight_right
            self.bias[i]        -= self.learn_r * delta_bias_right
            
    def predict(self,evl_batch_x):
        ''' Takes in data to be evaluated with the identified weights
            and biases from the neural network, gives the prediction as output '''
        # Create new arrays for the prediction case
        self.zp           = self.arrays()
        self.active_zp    = self.arrays()
        # Activate neurons for input layer
        self.zp[0]        = np.dot(evl_batch_x,self.weights[0]) + self.bias[0]
        self.active_zp[0] = self.act_I(self.zp[0])
        # Activate neurons for other layers
        for i in range(1,self.hidden_layers+1):
            self.zp[i]        = np.dot(self.active_zp[i-1],self.weights[i]) + self.bias[i]
            self.active_zp[i] = self.act_H(self.zp[i])
        # Activate neurons for output layer
        self.z[-1]        = np.dot(self.active_z[-2],self.weights[-1]) + self.bias[-1]
        self.active_z[-1] = self.act_O(self.z[-1])
        return(self.active_zp[-1],self.weights, self.bias)
            
    def cost_epoch(self):
        ''' Calculatest he cost for each epoch '''
        if self.costfunc == 1:
            # Cross entropy
            term_1 = self.batch_y*np.log(self.active_z[-1])
            term_2 = ((1-self.batch_y)*(np.log(1-self.active_z[-1])))
            cost = -np.sum(term_1+term_2)/len(self.batch_y)
        else:
            # Quadratic
            cost = 0.5*np.sum((self.active_z[-1]-self.batch_y)**2)
        return(cost)  
            
    def mini_batch_train(self,x,y,epochs,batch_size):
        ''' Perfrom mini-batch stochastic gradient descent. Inputs are
        training data and parameters, outputs are average cost'''
        nr_inputs          = len(y)
        batches            = int(nr_inputs/batch_size)
        self.batch_size = batch_size
        self.biases_and_weights(x)
        self.avgcost_epoch = np.zeros(epochs)
        for i in range(epochs):
            # Create batches
            batchlist_x, batchlist_y, batches = logmethods.batch(x,y,batch_size)
            #self.cost_epoch = np.zeros(batches)
            for j in range(batches):
                self.batch_x = np.asarray(batchlist_x[j])
                self.batch_y = np.asarray(batchlist_y[j])
                self.feed_forward()
                self.back_prop()
            self.avgcost_epoch[i] = self.cost_epoch() / nr_inputs
            print('Average_cost: ',i,': ',self.avgcost_epoch[i])
        return(self.avgcost_epoch)