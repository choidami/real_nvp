"""
The core Real-NVP model
"""

import tensorflow as tf
import real_nvp.nn as nn

layers = []

def construct_model_spec():
    global layers
    num_scales = 2
    
    for scale in range(num_scales-1):
        layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_1' % scale))
        layers.append(nn.CouplingLayer('checkerboard1', name='Checkerboard%d_2' % scale))
        layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_3' % scale))
        #layers.append(nn.CouplingLayer('checkerboard1', name='Checkerboard%d_4' % scale))
        #layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_5' % scale))
        #layers.append(nn.CouplingLayer('checkerboard1', name='Checkerboard%d_6' % scale))
        layers.append(nn.SqueezingLayer(name='Squeeze%d' % scale))
        layers.append(nn.CouplingLayer('channel0', name='Channel%d_1' % scale))
        layers.append(nn.CouplingLayer('channel1', name='Channel%d_2' % scale))
        layers.append(nn.CouplingLayer('channel0', name='Channel%d_3' % scale))
        layers.append(nn.FactorOutLayer(scale, name='FactorOut%d' % scale))
        
    # final layer
    scale = num_scales-1
    layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_1' % scale))
    layers.append(nn.CouplingLayer('checkerboard1', name='Checkerboard%d_2' % scale))
    layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_3' % scale))
    layers.append(nn.CouplingLayer('checkerboard1', name='Checkerboard%d_4' % scale))
#    layers.append(nn.SqueezingLayer(name='Squeeze%d' % scale))
#    layers.append(nn.CouplingLayer('channel0', name='Channel%d_1' % scale))
#    layers.append(nn.CouplingLayer('channel1', name='Channel%d_2' % scale))
#    layers.append(nn.CouplingLayer('channel0', name='Channel%d_3' % scale))
    layers.append(nn.FactorOutLayer(scale, name='FactorOut%d' % scale))

    
# the final dimension of the latent space is recorded here
# so that it can be used for constructing the inverse model
final_latent_dimension = []
def model_spec(x):
    xs = nn.int_shape(x)
    sum_log_det_jacobians = tf.zeros(xs[0])
    
    # model logit instead of the x itself
    y=x
    alpha=1e-5
    y = y*(1-alpha) + alpha*0.5
    jac = tf.reduce_sum(-tf.log(y) - tf.log(1-y), [1,2,3])
    y = tf.log(y) - tf.log(1-y)
    sum_log_det_jacobians += jac
  
    if len(layers) == 0:
        construct_model_spec()
            
    # construct forward pass
    z = None
    jac = sum_log_det_jacobians
    
    # TODO change for CIFAR-10
    #y=x
    
    for layer in layers:
        y,jac,z = layer.forward_and_jacobian(y, jac, z)
    
    if z is None:
        z = y
    else:
        z = tf.concat([z,y], axis=3)

    # record dimension of the final variable
    global final_latent_dimension
    final_latent_dimension = nn.int_shape(z)
    
    return z,jac

def inv_model_spec(y):
    #construct inverse pass for sampling
    shape = final_latent_dimension
    z = tf.reshape(y, [-1, shape[1], shape[2], shape[3]])
    y = None
    for layer in reversed(layers):
        y,z = layer.backward(y,z)
    
    # inverse logit
    x = tf.reciprocal(1 + tf.exp(-y))
    #x=y
    return x