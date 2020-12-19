from numpy import *

def sigmoid(x):
    return 1 / (1+exp(-x))

def dsigmoid(x):
    return x * (1 - x)

class NeuralNet:
    def __init__(self):
        self.hiddenLayerW = None
        self.outputLayerW = None
        self.output = None
        self.MSE = None
        self.trained = False
        
    def predict( self, X ):
        ### ... YOU FILL IN THIS CODE ....
        X = array([X])  # X was an array for some reason causing errors with X.shape, had to turn it into a matrix? But only one element? Dont know how this was supposed to work

        a0 = hstack((array([[1]*X.shape[0]]).T,X))  # Activation of input layer

        return sigmoid(dot(sigmoid(dot(a0, self.hiddenLayerW)), self.outputLayerW))[0]  # Computer activation of output layer
        
    def train(self,X,Y,hiddenLayerSize,epochs):    
        ## size of input layer (number of inputs plus bias)
        ni = X.shape[1] + 1

        ## size of hidden layer (number of hidden nodes plus bias)
        nh = hiddenLayerSize + 1

        # size of output layer
        no = 10

        ## initialize weight matrix for hidden layer
        self.hiddenLayerW = 2*random.random((ni,nh)) - 1
        ## initialize weight matrix for output layer
        self.outputLayerW = 2*random.random((nh,no)) - 1

        ## learning rate
        alpha = 0.001

        ## Mark as not trained
        self.trained = False
        ## Set up MSE array
        self.MSE = [0]*epochs

        for epoch in range(epochs):

            ### ... YOU FILL IN THIS CODE ....

            a0 = hstack((array([[1]*X.shape[0]]).T,X))  # Activation of input layer

            in0 = dot(a0, self.hiddenLayerW)            # Input to hidden layer

            a1 = sigmoid(in0)                           # Activation of hidden layer

            a1[:,0] = 1                                 # Set bias unit

            in1 = dot(a1, self.outputLayerW)            # Input to output layer

            a2 = sigmoid(in1)                           # Activation of output layer

            error_out = Y - a2                          # Observered error on output

            delta_out = error_out * dsigmoid(a2)        # Direction of targer

            ## Record MSE
            self.MSE[epoch] = mean(list(map(lambda x:x**2,error_out)))

            ### ... YOU FILL IN THIS CODE ...

            error_hidden = dot(delta_out, (self.outputLayerW).T)    # Contribution of hidden to error

            delta_hidden = error_hidden * dsigmoid(a1)  # Direction of target for hidden layer 

            self.hiddenLayerW = self.hiddenLayerW + dot(dot(alpha, a0.T), delta_hidden) # Hidden layer weight update

            self.outputLayerW = self.outputLayerW + dot(dot(alpha, a1.T), delta_out)    # Output layer weight update

        ## Update trained flag
        self.trained = True

