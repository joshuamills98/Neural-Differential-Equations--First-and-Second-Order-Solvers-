import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import time 
from keras.optimizers import Adam
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras import Model
tf.keras.backend.set_floatx('float32')
plt.style.use('fivethirtyeight')

#Here we define the basic neural network structure, without any physics information
class Block(Layer):
    def __init__(self):
        super(Block, self).__init__()
        self.linear_1 = Dense(5,activation = 'tanh')
        self.linear_2 = Dense(5,activation = 'tanh')

        self.linear_4 = Dense(1)

    def call(self,inputs, training=True):
        x = self.linear_1(inputs)
        x = self.linear_2(inputs)
        return self.linear_4(x)

class NeuralModel(Model):
    def __init__(self,u_0):
        super(NeuralModel,self).__init__()
        self.block = Block()
        self.u_0= u_0
        # self.es = EarlyStopping(monitor='train_loss',mode='min',verbose=1,patience=200,restore_best_weights=True)

    def call(self, inputs, training =True):
        if training:
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(inputs)
                x = self.block(inputs)
                x = self.u_0 + tf.math.multiply(inputs,x) #Implementing approximation to ODE g(t) = u_0 +NN(t)*t
            x = tape.gradient(x,inputs) #find dg/dt
        else:
            x = self.block(inputs)
            x = self.u_0+tf.math.multiply(inputs,x)
        return x

class ODESolver(NeuralModel): #In this class we put all the data particular to our equation
    def __init__(self, t,u_0):
        self.u_0=u_0
        super(ODESolver,self).__init__(u_0=self.u_0)
        self.t = t
        self.f = self.ODEFunction(t)

    def solve(self, epochs =400, verbose = 1):
        opt = Adam(learning_rate=0.01)
        self.compile(optimizer = opt, loss ='mean_squared_error')
        self.fit(self.t,self.f,epochs=400,verbose=verbose )

    def ODEFunction(self,t): #Define the function f(x) for the solver to output dy/dx = f(x)
        return(-2*t + np.cos(t))    


if __name__ == '__main__':
    u_0 = 2
    t = np.linspace(0,2,200)
    n_epochs = 1000
    ode = ODESolver(t,u_0=u_0)
    ode.compile(optimizer = 'Adam', loss ='mean_squared_error')
    fig,ax = plt.subplots(1,2, figsize=(15,8))
    ax[0].plot(t,-t**2+np.sin(t)+2,c='b', label = 'True Result')
    ax[0].plot(t,ode.predict(t),c='r',linestyle = '--',label = 'Predicted Result')
    ax[0].set_xlabel('x', fontsize = 24)
    ax[0].set_ylabel('y', fontsize = 24)
    ax[0].legend()
    ax[0].set_title('Before Training')
    print('Before Training Loss = {:.4f}'.format(ode.evaluate(t,-t**2+np.sin(t)+2)))

    start = time.time()
    ode.solve(epochs = n_epochs, verbose=0)
    end = time.time()

    ax[1].plot(t,-t**2+np.sin(t)+2,c='b', label = 'True Result')
    ax[1].plot(t,ode.predict(t),c='r',linestyle = '--',label = 'Predicted Result')
    ax[1].legend()
    ax[1].set_xlabel('x', fontsize = 24)
    ax[1].set_ylabel('y', fontsize = 24)
    ax[1].set_title('After Training')
    ax[1].legend()
    print('Loss After {} Epochs = {:.2f}. Total Run-Time = {:.2f}'.format(n_epochs, ode.evaluate(t,-t**2+np.sin(t)+2),end-start))
    plt.savefig('FirstOrderPINNExample.jpg', dpi = 300, transparent = True)
