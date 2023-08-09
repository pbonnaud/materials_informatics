#!/usr/bin/env python
# coding: utf-8
###     
### Last modification (DD/MM/YYY) : 07/08/2023
###
####################################################################################################
###                                                                                              ###
### Physics-Informed Neural Networks (PINNs): Basic example                                      ###
###                                                                                              ###
####################################################################################################

####################################################################################################
#                                                                                                  #
# Machine Learning for Materials Informatics                                                       #
#                                                                                                  #
# #### Markus J. Buehler, MIT                                                                      #
#                                                                                                  #
####################################################################################################

####################################################################################################
#                                                                                                  #
# https://towardsdatascience.com/solving-differential-equations-with-neural-networks-afdcf7b8bcc4  #
#                                                                                                  #
# https://github.com/madagra/basic-pinn                                                            #
#                                                                                                  #
####################################################################################################

### Import libraries ###############################################################################

import sys;

from torch import nn;

import torch;

from typing import Callable;

from scipy.integrate import solve_ivp;

import matplotlib.pyplot as plt;

import numpy as np;

from functools import partial;

from scipy.integrate import solve_ivp;

#sys.exit();

####################################################################################################
#                                                                                                  #
# Let's start by defining our neural network approximation function                                #
#                                                                                                  #
####################################################################################################

### Set the class for the neural network approximator ##############################################

class NNApproximator(nn.Module):

    # Simple neural network accepting one feature as input and returning a single      ***
    # output                                                                           ***
    #                                                                                  ***
    # In the context of PINNs, the neural network is used as universal function        ***
    # approximator to approximate the solution of the differential equation            ***
    
    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh()):

        super().__init__()

        print('num_hidden : {}'.format(num_hidden));

        print(' ');

        print('dim_hidden : {}'.format(dim_hidden)); 

        print(' ');

        self.layer_in = nn.Linear(1, dim_hidden);
        
        self.layer_out = nn.Linear(dim_hidden, 1);

        num_middle = num_hidden - 1;

        self.middle_layers = nn.ModuleList(
                             [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
                             );
        self.act = act;
        
        #self.dropout = nn.Dropout(0.25)

    def forward(self, x):

        #print (x.shape)

        out = self.act(self.layer_in(x));

        #print (out.shape)

        for layer in self.middle_layers:

            out = self.act(layer(out));

            #x = self.dropout(x)
            
        return self.layer_out(out);



nn_approximator = NNApproximator(10, 10);

#nn_approximator

#sys.exit();

### Get the number of parameters in the NN model ###################################################

x = sum(p.numel() for p in nn_approximator.parameters() if p.requires_grad);

print ("Parameters = ", x);

print(' ');    

#sys.exit();

####################################################################################################
#                                                                                                  #
# Now that we defined our universal function approximator, let’s build the loss function.          #
#                                                                                                  # 
# Two contributions - DE residial term. and BC/IC term                                             #
#                                                                                                  #
####################################################################################################

####################################################################################################
#                                                                                                  #
# Define loss functions:                                                                           #
# ----------------------                                                                           #
#                                                                                                  #
# Before we get started, some basics on automatic differention in PyTorch                          #
#                                                                                                  #
# See also: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html                    #
#                                                                                                  #
####################################################################################################

# Define tensors x and y ///////////////////////////////////////////////////////////////////////////
                                                                                         #
# requires_grad=True: signals to autograd that every operation on them should be       *** 
#                     tracked.                                                         ***

x = torch.tensor(2., requires_grad=True);

y = torch.tensor(3., requires_grad=True);

# Define function z ////////////////////////////////////////////////////////////////////////////////

z = x * x * y;

print('The z function is z = x^2 * y with x = 2 and y = 3');

print(' ');

# Compute the gradient of z with respect to all the tensors in the computation graph ///////////////

z.backward();

# Print the gradient of z with respect to x and y //////////////////////////////////////////////////

print('The gradient of z with respect to x is (z = 2 * x * y) : {}'.format(x.grad));

print(' ');

print('The gradient of z with respect to y is (z = x^2) : {}'.format(y.grad));

print(' ');

#sys.exit();

### Use of the autograd.grad function available in PyTorch #########################################
                                                                                         #
# autograd.grad: computes and returns the sum of gradients of outputs with respect to  ***
# the inputs.                                                                          ***
                                                                                         #
x = torch.tensor(2.).requires_grad_();

y = torch.tensor(3.).requires_grad_();

z = x * x * y;

grad_x = torch.autograd.grad(outputs=z, inputs=x);

print(grad_x[0]);

#sys.exit();

z = x * x * y;

grad_y = torch.autograd.grad(outputs=z, inputs=y);

print(grad_y[0]);

#sys.exit();

####################################################################################################
#                                                                                                  #
#  Definition of losses:                                                                           #
#  ---------------------                                                                           #
#                                                                                                  #
####################################################################################################

# Function f that conmputes the value of the approximate solution from the NN model #################

def f(nn: NNApproximator, x: torch.Tensor) -> torch.Tensor:

    return nn(x);

# Function df that computes the NN derivative with respect to inputs using autograd ################

def df(nn: NNApproximator, x: torch.Tensor = None, order: int = 1) -> torch.Tensor:

    df_value = f(nn, x);

    for _ in range(order):

        df_value = torch.autograd.grad(df_value,
                                       x,
                                       grad_outputs=torch.ones_like(x),
                                       create_graph=True,
                                       retain_graph=True,
                                      )[0]

    return df_value

# NOTE: While it is not needed here, a repeated application of the torch.autograd.grad ***
#       function can compute any arbitrary order derivatives.                          ***

####################################################################################################
#                                                                                                  #
# Definition of the total loss:                                                                    #
# -----------------------------                                                                    #
#                                                                                                  #
# * Determination of colocation points                                                             #
#                                                                                                  #
# * Begin with linear spacing                                                                      #
#                                                                                                  #
# NOTE: For more complex problems, the choice of colocation points is very important and requires  # 
#       a much more careful choice.                                                                #
#                                                                                                  #
####################################################################################################

### Set the domain of interest #####################################################################

domain = [0.0, 1.0];

### Apply the linear spacing of colocation points ##################################################

t = torch.linspace(domain[0], domain[1], steps=10, requires_grad=True);

### Reshape the list of colocation points ##########################################################

print ('Initial shape : ', t.shape);

print(' ');

t = t.reshape(t.shape[0], 1);

print ('New shape : ', t.shape);

print(' ');

#sys.exit();

#nn_approximator

#t

#nn_approximator(t);

# Using the functions above, the MSE loss is easily computed as a sum of the DE        ***
# contribution at each colocation point and the boundary contribution:                 ***

# ![image.png](attachment:image.png)

### Set parameters and boundary conditions #############################################3###########

R  = 1;           # parameter in DE

T0 = 0.0;         # initial time 

F0 = 1.0;         # boundary condition value

# Set the contribution from the DE (Differential Equation) /////////////////////////////////////////
 
interior_loss = df(nn_approximator, t) - R * t * (1 - t);

# Set boundary contributions ///////////////////////////////////////////////////////////////////////

boundary = torch.Tensor([T0]);

boundary.requires_grad = True;

boundary_loss = f(nn_approximator, boundary) - F0;

# ![image.png](attachment:image.png)

# Set the final loss ///////////////////////////////////////////////////////////////////////////////

final_loss_L = interior_loss.pow(2).mean() + boundary_loss**2   

# The average over all the colocation points + boundary contribution is just a single value ////////

print('final_loss_L : ', final_loss_L); 

print(' ');

#sys.exit();

####################################################################################################
#                                                                                                  #
# Wrap the loss calculation in a nice sub                                                          #
#                                                                                                  #
####################################################################################################

# Function that computes the full loss function ####################################################
                                                                                          
def compute_loss(nn: NNApproximator, 
                 x: torch.Tensor = None, 
                 verbose: bool = False) -> torch.float:

    # Compute the full loss function as interior loss + boundary loss                  ***
    # This custom loss function is fully defined with differentiable tensors           ***
    # therefore the .backward() method can be applied to it                            ***

    interior_loss = df(nn, x) - R * x * (1 - x);

    # Bondary conditions ///////////////////////////////////////////////////////////////////////////

    boundary = torch.Tensor([0.0]);

    boundary.requires_grad = True;

    boundary_loss = f(nn, boundary) - F0;

    # Set the final (total) loss ///////////////////////////////////////////////////////////////////
    
    final_loss = interior_loss.pow(2).mean() + boundary_loss**2;

    return final_loss;

####################################################################################################
#                                                                                                  #
# Construction of a training loop and optimization                                                 # 
#                                                                                                  #
####################################################################################################

f_initial = f(nn_approximator, t);

fig, ax = plt.subplots();

ax.plot(t.detach().numpy(), f_initial.detach().numpy(), label="Initial NN solution");

# Function that does the training loop #############################################################

def train_model(nn: NNApproximator,
                loss_fn: Callable,
                learning_rate: int = 0.01,
                max_epochs: int = 1_000,
               ) -> NNApproximator:

    loss_evolution = [];

    optimizer = torch.optim.SGD(nn.parameters(), lr=learning_rate);

    for epoch in range(max_epochs):

        try:

            loss: torch.Tensor = loss_fn(nn);

            optimizer.zero_grad();

            loss.backward();

            optimizer.step();

            if epoch % 1000 == 0:

                print(f"Epoch: {epoch} - Loss: {float(loss):>7f}");

            loss_evolution.append(loss.detach().numpy());

        except KeyboardInterrupt:

            break;

    return nn, np.array(loss_evolution)

### Make compute_loss callable with one variable ###################################################

loss_fn = partial(compute_loss, x=t, verbose=True);

### Train the model ################################################################################

nn_approximator_trained, loss_evolution = train_model(nn_approximator,
                                                      loss_fn=loss_fn, 
                                                      learning_rate=0.2, 
                                                      max_epochs=10_000
                                                     );

### Creation of a sequence of 100 evenly spaced points between domain[0] and domain[1] #############

# x_eval is a column vector

x_eval = torch.linspace(domain[0], domain[1], steps=100).reshape(-1, 1);

# Function that develops the ground truth numeric solution (analytical solution) ###################

def logistic_eq_fn(x, y):

        return R * x * (1 - x);

### Generate the analytical solution ###############################################################

numeric_solution = solve_ivp(logistic_eq_fn, 
                             domain,        
                             [F0], 
                             t_eval=x_eval.squeeze().detach().numpy()
                            );

####################################################################################################
#                                                                                                  #
# Show results                                                                                     #
#                                                                                                  #
####################################################################################################

# Function that shows the results ##################################################################

def show_results (t, numeric_solution, x_eval):

    # plotting data ////////////////////////////////////////////////////////////////////////////////

    fig, ax = plt.subplots();

    f_final_training = f(nn_approximator_trained, t);

    f_final = f(nn_approximator_trained, x_eval);

    ax.scatter(t.detach().numpy(), f_final_training.detach().numpy(), label="Training points", color="red");

    ax.plot(x_eval.detach().numpy(), f_final.detach().numpy(), label="NN final solution");

    ax.plot(x_eval.detach().numpy(),
            numeric_solution.y.T,
            label=f"Analytic solution",
            color="green",
            alpha=0.75,
           );

    ax.set(title="Logistic equation solved with NNs", xlabel="t", ylabel="f(t)");

    ax.legend();

    fig, ax = plt.subplots();

    ax.semilogy(loss_evolution);

    ax.set(title="Loss evolution", xlabel="# epochs", ylabel="Loss");

    ax.legend();

    plt.show();

### Apply the function for showing the results #####################################################

show_results(t, numeric_solution, x_eval);

####################################################################################################
#                                                                                                  #
# Same model with different initial condition, different domain                                    #
#                                                                                                  #
####################################################################################################

domain = [0.0, 1.0];

t = torch.linspace(domain[0], domain[1], steps=10, requires_grad=True);

t = t.reshape(t.shape[0], 1);

print('The shape of t is {}'.format(t.shape));

print(' ');

### Set new initial conditions #####################################################################

R = 2;             # parameter in DE

T0 = 0.0;          # initial time
 
F0 = -1.0;         # boundary condition value

### Training of the NN model with the new boundary conditions ######################################

nn_approximator_trained, loss_evolution = train_model(nn_approximator, 
                                                      loss_fn=loss_fn, 
                                                      learning_rate=0.1, 
                                                      max_epochs=10_000
                                                     );


### Creation of a sequence of 100 evenly spaced points between domain[0] and domain[1] #############
            
# x_eval is a column vector

x_eval = torch.linspace(domain[0], domain[1], steps=100).reshape(-1, 1);

    
### Generate the analytical solution ###############################################################
    
numeric_solution = solve_ivp(logistic_eq_fn, 
                             domain, 
                             [F0], 
                             t_eval=x_eval.squeeze().detach().numpy()
                            );

### Show the resulst with the new conditions #######################################################

show_results(t, numeric_solution, x_eval);

