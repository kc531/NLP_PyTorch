import torch
import torch.nn as nn
from torch.optim import Optimizer
import math

class AdamCustom(Optimizer):
  def __init__(self, params, lr = 0.001, betas = (0.9,0.99), eps = 1e-8, weight_decay = 0):
    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    super(AdamCustom, self).__init__(params, defaults)
  
  def step(self):
    #param_groups is inheritered which is used to break model parameters to seprate components for optimisation
    #helps in training seperate layers of network
    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data
        #In gerneral adam doesn't support sparse gradients
        #if grad.is_sparse:
	      #  raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
        state = self.state[p]
        if len(state) == 0:
          state['step'] = 0
          # Momentum (Exponential MA of gradients)
          state['exp_avg'] = torch.zeros_like(p.data)
          #print(p.data.size())
          # RMS Prop componenet. (Exponential MA of squared gradients). Denominator.
          state['exp_avg_sq'] = torch.zeros_like(p.data)

        momentum, rmsprop = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']
        state['step'] += 1
        #exponential weighted average calculation
        momentum = torch.mul(momentum, beta1) + (1-beta1)*grad
        rmsprop = torch.mul(rmsprop, beta2) + (1-beta2)*(grad**2)
        
        denom = rmsprop.sqrt() + group['eps']

        #bias correction for exponential moving average
        #if group['correct_bias']:
        alpha  = group['lr']
        bias_correction1 = 1 / (1 - beta1 ** state['step'])
        bias_correction2 = 1 / (1 - beta2 ** state['step'])

        #New Learning Rate
        adapted_lr = alpha * bias_correction1 / math.sqrt(bias_correction2)

        p.data = p.data - adapted_lr * (momentum/denom)

        #weight decay helps in adding a penality term to the cost function 
        #which has the effect of shrinking the weights during backpropagation. 
        #This helps prevent the network from overfitting the training data as well as the exploding gradient problem
        #if group['weight_decay'] > 0.0:
	      #  p.data.add_(-group['lr'] * group['weight_decay'], p.data)
