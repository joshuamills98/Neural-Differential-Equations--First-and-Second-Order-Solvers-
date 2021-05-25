import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import tensorflow as tf
from tensorflow.keras.layers import RNN, Layer, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow import float32, concat, convert_to_tensor, linalg
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
import time 
tf.keras.backend.set_floatx('float32')
plt.style.use('fivethirtyeight')

#Here we define the basic neural network structure, without any physics information
class Block(Layer):
    def __init__(self):
        super(Block, self).__init__()
        self.linear_1 = Dense(32,activation = 'tanh',kernel_initializer = 'glorot_uniform', bias_initializer = 'zeros')
        self.linear_2 = Dense(32,activation = 'tanh',kernel_initializer = 'glorot_uniform', bias_initializer = 'zeros')
        self.linear_3 = Dense(32,activation = 'tanh',kernel_initializer = 'glorot_uniform', bias_initializer = 'zeros')
        self.linear_4 = Dense(1)

    def call(self,inputs, training=True):
        x = self.linear_1(inputs)
        x = self.linear_2(x)
        x = self.linear_3(x)
        return self.linear_4(x)

#We define a NeuralModel to predict d^2ydx^2 = f(x)
class NeuralModel(Model):
    def __init__(self):
        super(NeuralModel,self).__init__()
        self.block = Block()
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs, training =True):
        if training:
            with tf.GradientTape(persistent=True) as g:
                g.watch(inputs)
                with tf.GradientTape(persistent=True) as gg:
                    gg.watch(inputs)
                    y = self.block(inputs)
                dy_dx = gg.gradient(y,inputs)  
            d2y_dx2 = g.gradient(dy_dx,inputs) 
            return (y[0], dy_dx[0], d2y_dx2) #Return predicted boundary conditions and predicted f(x) as a tuple

        else:
            y = self.block(inputs) #Return the output y 
            return y

    def custom_loss(self, y_true, y_pred):
        y0_true, dy_dx0_true, fx = y_true #Unpack the true values
        y0_pred, dy_dx0_pred ,d2y_dx2_pred = y_pred #Unpack the predicted values
        self.loss = tf.reduce_mean(tf.math.square(d2y_dx2_pred-fx))+tf.reduce_mean(tf.math.square(y0_pred-y0_true))+tf.reduce_mean(tf.math.square(dy_dx0_pred-dy_dx0_true)) 
        return self.loss

    def train_step(self,data): #Overwright Keras training step to suit ODE Model
        x,y_true = data #Pass to the NN input domain, f(x), and the two boundary conditions y is a tuple
        with tf.GradientTape() as tape:
            prediction = self(x,training=True)
            self.loss = self.custom_loss(y_true,prediction)
        
        #Compute gradients:
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(self.loss,trainable_vars)

        self.optimizer.apply_gradients(zip(gradients,trainable_vars))
        self.loss_tracker.update_state(self.loss)
        return {"loss": self.loss_tracker.result()}

class ODESolver(NeuralModel): 
    """
    Here the ODESolver is defined: User input required is the ODEFunction
    """
    def __init__(self,y0,dy_dx0,x,learning_rate):
        super(ODESolver,self).__init__()
        self.y0 = y0 #First Boundary Condition
        self.dy_dx0 = dy_dx0 #2nd Boundary Condition
        self.x = x #Domain
        self.n = x.shape[0]
        self.y_true = self.data_preparation(self.y0, self.dy_dx0, self.x)
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.compile(optimizer = self.opt, loss  = self.custom_loss)

    def data_preparation(self, y0, dy_dx0, x):
        y0 = np.repeat(y0,self.n).reshape(-1,1)
        dy_dx0 = np.repeat(dy_dx0,self.n).reshape(-1,1)
        fx = self.ODEFunction(x)
        return (y0,dy_dx0,fx)

    def solve(self, epochs =400, batch_size=1, verbose = 1): # Custom solver since using .fit not working
        for i in range(epochs):
            loss = self.train_step([self.x,self.y_true])
            if i %100 == 0:
                print('Epoch = {}. Loss = {}'.format(i, loss['loss']))

    def ODEFunction(self,x): #Define the function f(x) for the solver to output d^2y/dx^2 = f(x)
        return(-np.sin(x)+x)    

if __name__ == '__main__':
    n_epochs = 500
    x=np.linspace(0,np.pi,100, dtype=np.float32)
    x = np.reshape(x,(-1,1))
    ode = ODESolver(0.,1.,x,0.02)
    fig, ax = plt.subplots(1,2, figsize = (12,6))
    ax[0].plot(x,np.sin(x)+x**3/6,c='b', label = 'True Output')
    ax[0].plot(x,ode.predict(x), c='r', linestyle = '--', label = 'Predicted Output')
    ax[0].set_title('Before Training')
    ax[0].legend()
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')

    start = time.time()
    ode.solve(epochs=n_epochs)
    end = time.time()
    ax[1].plot(x,np.sin(x)+x**3/6,c='b', label = 'True Output')
    ax[1].plot(x,ode.predict(x), c = 'r',linestyle = '--', label = 'Predicted Output')
    ax[1].set_title('After Training')
    ax[1].legend()
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    print('Loss After {} Epochs = {:.2f}. Total Run-Time = {:.2f}'.format(n_epochs, ode.loss.numpy(),end-start))
    plt.savefig('SecondOrderPINNExample.jpg')
