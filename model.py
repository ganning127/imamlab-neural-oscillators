from torch import nn
import torch
from torch.autograd import Variable
import math
from torch.distributions.uniform import Uniform
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import sys



class coRNNCell(nn.Module):
    def __init__(self, network_type, n_inp, n_hid, dt, gamma, epsilon):
        super(coRNNCell, self).__init__()

        #network parameters
        self.network_type = network_type
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon


        #define activation fxn
        self.activation = nn.Tanh()

        #input weights
        self.I_ext = nn.Linear(in_features=n_inp, out_features=n_hid, 
                          bias=True)
        
        #recurrent weights of hidden states
        self.R = nn.Linear(in_features=n_hid, out_features=n_hid, 
                      bias = False)

        #recurrent weights of velocity of hidden states
        self.F = nn.Linear(in_features = n_hid, out_features=n_hid, 
                      bias = False)
        


    def forward(self,x,hy,hz):

        #our update equations
        activation = self.activation(self.R(hy) + self.F(hz) + 
                                          self.I_ext(x))

        hz = hz + self.dt*(activation- self.gamma*hy - self.epsilon*hz)
    

        hy = hy + self.dt*hz
        
        return hy, hz, activation
    


class coRNN(nn.Module):
    def __init__(self, network_type, n_inp, n_hid, n_out, dt, gamma, epsilon):
        super(coRNN, self).__init__()

        #network parameters
        self.n_hid = n_hid
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_out = n_out

        self.cell = coRNNCell(network_type, n_inp, n_hid, dt, self.gamma, self.epsilon)
        self.readout = nn.Linear(n_hid, n_out)
        
    
    def forward(self, x):

        ## initialize hidden states
        hy = Variable(torch.zeros(x.size(1),self.n_hid, device=device))
        hz = Variable(torch.zeros(x.size(1),self.n_hid, device=device))
        

        #save hidden state values
        save_hy = torch.zeros(size=(x.size(0), x.size(1), self.n_hid), requires_grad=False, device = device) #(input_size, batch, n_hid)
        save_hz = torch.zeros(size=(x.size(0), x.size(1), self.n_hid), requires_grad=False, device = device)
        save_activation = torch.zeros(size=(x.size(0), x.size(1), self.n_hid), requires_grad=False, device = device)
        save_output = torch.zeros(size=(x.size(0), x.size(1), self.n_out), requires_grad=False, device = device)

        for t in range(x.size(0)):
            hy, hz, activation = self.cell(x[t],hy,hz)


            save_hy[t, :, :] = hy
            save_hz[t, :, :] = hz
            save_activation[t, :, :] = activation
            save_output[t, :, :] = self.readout(hy)


        return save_output, save_hy, save_hz, save_activation
