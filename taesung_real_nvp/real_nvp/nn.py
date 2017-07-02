import numpy as np
import tensorflow as tf

def int_shape(x):
    return list(map(int, x.get_shape()))

# Abstract class that can do both the forward and backward pass
class Layer():
    def forward_and_jacobian(self, x, sum_log_det_jacobians, z):
        raise NotImplementedError(str(type(self)))
        
    def backward(self, y, z):
        raise NotImplementedError(str(type(self)))
        
# The coupling layer
# Contains code for both checkerboard and channelwise masking.
class CouplingLayer(Layer):
    
    # |mask_type| can be 'checkerboard0', 'checkerboard1', 'channel0', 'channel1'
    def __init__(self, mask_type, name='Coupling'):
        self.mask_type = mask_type
        self.name = name
        
    # Batch normalization 
    # Need to implement Moving average batch normalization 
    def batch_norm(self, x):
        mu = tf.reduce_mean(x)
        sig2 = tf.reduce_mean(tf.square(x-mu))
        x = (x-mu)/tf.sqrt(sig2 + 1.0e-6)
        return x, sig2
    
    # Weight normalization 
    def get_normalized_weights(self, name, weights_shape):
        weights = tf.get_variable(name, weights_shape, tf.float32,
                                  tf.contrib.layers.xavier_initializer())
        scale = tf.get_variable(name + "_scale", [1], tf.float32,
                                tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(5e-5))
        norm = tf.sqrt(tf.reduce_sum(tf.square(weights)))
        return weights/norm * scale
    
    # convnets that represent s and t 
    def function_s_t(self, x, mask, name='function_s_t'):
        with tf.variable_scope(name):
            channel = 64
            padding = 'SAME'
            xs = int_shape(x)
            kernel_h = 3
            kernel_w = 3
            input_channel=xs[3]
            y = x

            y,_ = self.batch_norm(y)
            weights_shape = [1, 1, input_channel, channel]
            weights = self.get_normalized_weights("weights_input", weights_shape)
            
            y = tf.nn.conv2d(y, weights, [1,1,1,1], padding=padding)
            y,_ = self.batch_norm(y)
            y = tf.nn.relu(y)
    
            skip = y
            # Residual blocks
            num_residual_blocks = 5
            for r in range(num_residual_blocks):
                weights_shape = [kernel_h, kernel_w, channel, channel]
                weights = self.get_normalized_weights("weights%d_1" %r, weights_shape)
                y = tf.nn.conv2d(y, weights, [1,1,1,1], padding=padding)
                y,_ = self.batch_norm(y)
                y = tf.nn.relu(y)
                
                weights = self.get_normalized_weights("weights%d_2" %r, weights_shape)
                y = tf.nn.conv2d(y, weights, [1,1,1,1], padding=padding)
                y,_ = self.batch_norm(y)
                y += skip
                y = tf.nn.relu(y)
                skip = y
                
            # 1x1 convolution for reducing dimension
            # the *2 factor on the output channel is because of s and t
            weights = self.get_normalized_weights("weights_output",
                                                  [1,1,channel, input_channel*2])
            y = tf.nn.conv2d(y, weights, [1,1,1,1], padding=padding)
            
            # For numerical stabiilty apply tanh and then scale
            # TODO: Try to only apply tanh to s not t
            s = y[:,:,:,:input_channel]
            t = y[:,:,:,input_channel:]
            
            s = tf.tanh(s)
            scale_factor = self.get_normalized_weights("weights_tanh_scale", [1])
            s *= scale_factor
            
            # The first half defines the s function
            # The second half defines the t function
            s = s * (-mask+1)
            t = t * (-mask+1)
            
        return s, t
            
            
    def get_mask(self, xs, mask_type):
        ''' returns constant tensor as mask
            xs: size of tensor
            mask_type: one of 'checkerboard0','checkerboard1','channel0','channel1'
            b: output mask-has the dimension of xs'''
        if 'checkerboard' in mask_type:
            unit0 = tf.constant([[0.0, 1.0], [1.0, 0.0]])
            unit1 = -unit0 + 1
            unit = unit0 if mask_type == 'checkerboard0' else unit1
            unit = tf.reshape(unit, [1,2,2,1]) #[batch_size, w, h, channel_num]
            b = tf.tile(unit, [xs[0], xs[1]//2, xs[2]//2, xs[3]])
        elif 'channel' in mask_type:
            white = tf.ones([xs[0], xs[1], xs[2], xs[3]//2])
            black = tf.zeros([xs[0], xs[1], xs[2], xs[3]//2])
            if mask_type == 'channel0':
                b = tf.concat([white,black], axis=3)
            else:
                b = tf.concat([black, white], axis=3)
                
        bs = int_shape(b)
        assert bs == xs
        
        return b
    
    def forward_and_jacobian(self, x, sum_log_det_jacobians, z, reuse=False):
        ''' Coupling Layer that goes from x -> z
            TODO: Potentially remove z from arg
                  Add reuse in the argument'''
        with tf.variable_scope(self.name, reuse=reuse):
            xs = int_shape(x)
            b = self.get_mask(xs, self.mask_type)
            
            x1 = x * b
            s,t = self.function_s_t(x1, b)
            y = x1 + tf.multiply(-b+1.0, x*tf.check_numerics(tf.exp(s), "exp has NaN") + t)
            log_det_jacobian = tf.reduce_sum(s, [1,2,3])
            sum_log_det_jacobians += log_det_jacobian
            
            return y, sum_log_det_jacobians, z
    
    def backward(self, y, z):
        with tf.variable_scope(self.name, reuse=True):
            ys = int_shape(y)
            b = self.get_mask(ys, self.mask_type)
            
            y1 = y * b
            s,t = self.function_s_t(y1, b)
            x = y1 + tf.multiply(y*(-b+1.0) - t, tf.exp(-s))
            return x,z
        
# The layer that performs squeezing. Only changes the dim
# The Jacobian is untouched and passed to the next layer            
class SqueezingLayer(Layer):
    def __init__(self, name="Squeeze"):
        self.name = name
        
    def forward_and_jacobian(self, x, sum_log_det_jacobians, z):
        xs = int_shape(x)
        assert xs[1]%2 == 0 and xs[2] % 2 == 0
        y = tf.space_to_depth(x,2)
        if z is not None:
            z = tf.space_to_depth(z, 2)
        
        return y, sum_log_det_jacobians, z
    
    def backward(self, y, z):
        ys = int_shape(y)
        assert ys[3] % 4 == 0
        x = tf.depth_to_space(y, 2)
        if z is not None:
            z = tf.depth_to_space(z,2)
        
        return x,z
 
# The layer that factors out half of the dimension
# maps them directly to the latent space (gaussian)
class FactorOutLayer(Layer):       
    def __init__(self, scale, name='FactorOut'):
        self.scale = scale
        self.name = name
        
    def forward_and_jacobian(self, x, sum_log_det_jacobians, z):
        xs = int_shape(x)
        split = xs[3]//2
        
        # The factoring out is done channel-wise
        # TODO: figure out how to do factoring non-channel-wise
        new_z = x[:,:,:,:split]
        x = x[:,:,:,split:]
        
        if z is not None:
            z = tf.concat([z, new_z], axis=3)
        else:
            z= new_z
        return x, sum_log_det_jacobians, z
    
    def backward(self, y, z):
        # At scale 0, 1/2 of the original dimensions are factored out
        # At scale 1, 1/4 of the original dimensions are factored out
        # ...
        # At scale s, (1/2)^(s+1) are factored out
        # Hence, at backward pass of scale s, (1/2)^s of z should be factored in

        zs = int_shape(z)
        if y is None:
            split = zs[3] // (2**self.scale)
        else:
            split = int_shape(y)[3]
        new_y = z[:,:,:,-split:]
        z = z[:,:,:,:-split]
        
        assert (int_shape(new_y)[3] == split)
        
        if y is not None:
            x = tf.concat([new_y, y], axis=3)
        else:
            x = new_y
        
        return x, z
    
def compute_log_prob_x(z, sum_log_det_jacobians):
    ''' Given the output of the network (z) and all sum_log_det_jacobians,
        compute the log-probability'''
    
    # z is assumed to be in standard normal distribution
    # 1/sqrt(2*pi)*exp(-0.5*x^2)
    
    zs = int_shape(z)
    K = zs[1]*zs[2]*zs[3] #dimension of the Gaussian distribution
    
    log_density_z = -0.5*(tf.reduce_sum(tf.square(z), [1,2,3]) + K*np.log(2*np.pi))
    
    log_density_x = log_density_z + sum_log_det_jacobians
    
    # to go from density to probability, one can 
    # multiply the density by the width of the 
    # discrete probability area, which is 1/256.0, per dimension.
    # The calcculation is performed in the log space.
    log_prob_x = log_density_x #- K*tf.log(256.0)
    
    return log_prob_x

def loss (z, sum_log_det_jacobians):
    ''' Computes the loss of the network
        It is chosen so that the probability P(x) of the 
        natural images is maximized.'''
    return -tf.reduce_sum(compute_log_prob_x(z, sum_log_det_jacobians))

def adam_updates(params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999):
    ''' Adam optimizer 
        Exactly the same code as the Pixel CNN++ implementation by OpenAI
        https://github.com/openai/pixel-cnn'''
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads
    t = tf.Variable(1., 'adam_t')
    for p, g in zip(params, grads):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
        if mom1>0:
            v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')
            v_t = mom1*v + (1. - mom1)*g
            v_hat = v_t / (1. - tf.pow(mom1,t))
            updates.append(v.assign(v_t))
        else:
            v_hat = g
        mg_t = mom2*mg + (1. - mom2)*tf.square(g)
        mg_hat = mg_t / (1. - tf.pow(mom2,t))
        g_t = v_hat / tf.sqrt(mg_hat + 1e-8)
        p_t = p - lr * g_t
        updates.append(mg.assign(mg_t))
        updates.append(p.assign(p_t))
    updates.append(t.assign_add(1))
    return tf.group(*updates)