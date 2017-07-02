import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
  
def gaussianDistribution(h):
    return -0.5*(np.square(h) + 784*np.log(2*np.pi))
    
def getMask(shape):
    dim = int(np.sqrt(shape[1]))
    return np.reshape([1 if (i+j)%2 else 0 for i in xrange(dim) for j in xrange(dim)]*shape[0], shape)

def get_normalized_weights(name, weights_shape):
    weights = tf.get_variable(name, weights_shape, tf.float32,
                              tf.contrib.layers.xavier_initializer())
    norm = tf.sqrt(tf.reduce_sum(tf.square(weights)))
    return weights/norm

def convnet(inputs, is_training, scope="convnet"):
    inputs = tf.reshape(inputs, [-1,28,28,1])
    with tf.variable_scope(scope, "convnet", [inputs]) as var_scope:
        with slim.arg_scope([slim.conv2d],
                            kernel_size=[5,5],
                            stride=1,
                            activation_fn=tf.nn.relu,
                            #normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training},
                            weights_regularizer=slim.l2_regularizer(0.0005)
                            ):
            net = slim.conv2d(inputs, 32, kernel_size=[5,5])
            net = slim.max_pool2d(net, 2, [2,2])
            net = slim.conv2d(net, 64, kernel_size=[5,5])
            net = slim.max_pool2d(net, 2, [2,2])
            net = slim.flatten(net)
            #net = slim.fully_connected(net, 1024, normalizer_fn=slim.batch_norm, weights_regularizer=slim.l2_regularizer(0.00005))
            #net = slim.fully_connected(net, 900, normalizer_fn=slim.batch_norm, weights_regularizer=slim.l2_regularizer(0.00005))
            #net = slim.fully_connected(net, 850, normalizer_fn=slim.batch_norm, weights_regularizer=slim.l2_regularizer(0.00005))
            #net = slim.fully_connected(net, 800, normalizer_fn=slim.batch_norm, weights_regularizer=slim.l2_regularizer(0.00005))
            
            #net = slim.fully_connected(net, 784, activation_fn=tf.tanh)
            if "/s/" in var_scope.name:
                net = slim.fully_connected(net, 784, activation_fn=tf.tanh, weights_regularizer=slim.l2_regularizer(0.00005), normalizer_params={'is_training': is_training})
                scale_factor = get_normalized_weights("weights_tanh_scale_s",[1])
                #print(scale_factor.eval())
            else:
                net = slim.fully_connected(net, 784, activation_fn=tf.tanh, weights_regularizer=slim.l2_regularizer(0.00005), normalizer_params={'is_training': is_training})
                scale_factor = get_normalized_weights("weights_tanh_scale_t",[1])
            return net*scale_factor

def forward_layer(x_input, variable_scope, mask, is_training):
    with tf.variable_scope(variable_scope):
        with tf.variable_scope("s"):
            scale = convnet(x_input*mask, is_training)
        with tf.variable_scope("t"):
            translation = convnet(x_input*mask, is_training)
    return (mask*x_input + (1-mask)*(x_input*tf.exp(scale) + translation), scale)

def backwards_layer(y_input, variable_scope, mask, is_training):
    with tf.variable_scope(variable_scope, reuse=True):
        with tf.variable_scope("s"):
            scale = convnet(y_input*mask, is_training)
        with tf.variable_scope("t"):
            translation = convnet(y_input*mask, is_training)
    return mask*y_input + (1-mask)*(y_input - translation)*tf.exp(-scale)

def forward_pass(x, layer_num, mask, is_training):
    scale = 0
    y = x
    for i in xrange(layer_num):
        if i%2:
            y, s = forward_layer(y, "layer"+str(i+1),(1-mask), is_training)
        else:
            y, s = forward_layer(y, "layer"+str(i+1) , mask, is_training)
        scale += s
    return y, scale
    
def backward_pass(y, layer_num, mask, is_training):
    #tf.get_variable_scope().reuse_variables()
    x = y    
    for i in xrange(layer_num-1, -1, -1):
        if i%2:
            x = backwards_layer(x, "layer"+str(i+1),(1-mask), is_training)
        else:
            x = backwards_layer(x, "layer"+str(i+1) , mask, is_training)
    return x
  
def drawMNISTs(digits): # plots MNIST from a [784, num_digits] array.
  plt.figure(1)
  plt.clf()
  for i in xrange(digits.shape[1]):
    plt.subplot(1, digits.shape[1], i+1)
    plt.imshow(digits[:, i].reshape(28, 28).T)
  plt.draw()
  raw_input('Press Enter.')
  
def drawMNIST(digit, title): # plots MNIST from a [784, num_digits] array.
  plt.figure(1)
  plt.clf()
  plt.title(title)
  plt.imshow(np.reshape(digit, (28, 28)))
  plt.title(title)
  plt.draw()
  raw_input('Press Enter.')


if __name__ == '__main__':
    slim = tf.contrib.slim
    
    layer_num = 4
    batch_size = 50
    num_epoch = 2000
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.InteractiveSession(config=config)    
    
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    x = tf.placeholder(tf.float32, shape=[None, 784])
    is_training = tf.placeholder(tf.bool)
    mask = tf.placeholder(tf.float32, shape=[None, 784])
    h, s = forward_pass(x, layer_num, mask, is_training)
    
    # Calculate log-likelihood
    log_likelihood = -tf.reduce_sum(gaussianDistribution(h) + s )#- 784*tf.log(256.0))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(
                learning_rate=0.001,
                beta1=1. - 1e-1,
                beta2=1. - 1e-3,
                epsilon=1e-08).minimize(log_likelihood)
    
    log_likelihoods = []
    
    sess.run(tf.global_variables_initializer())
    for i in range(num_epoch):
      batch = mnist.train.next_batch(batch_size)
      if i%100 == 0:
          loglikelihood = log_likelihood.eval(feed_dict={x: batch[0], mask: getMask((batch_size, 784)), is_training:False})
          log_likelihoods.append(loglikelihood)
          print("step %d, log-likelihood %g"%(i, loglikelihood))
          #output = h.eval(feed_dict={x: batch[0], mask: getMask((batch_size, 784))})[0]
          #plt.figure(i)
          #plt.clf()
          #plt.imshow(np.reshape(output, (28, 28)), cmap=plt.cm.gray)
          #plt.draw()
        
      train_step.run(feed_dict={x: batch[0], mask: getMask((batch_size, 784)), is_training:True})
    
    #get test-set log-likelihood
    #mask_test = getMask((len(mnist.test.images), 784))
    #print("log-likelihood %g"%log_likelihood.eval(feed_dict={x: mnist.test.images, mask: mask_test}))
    
    #plot log-likelihoods
    plt.clf()
    x_axis = np.linspace(0,num_epoch, num_epoch/100)    
    plt.plot(x_axis, log_likelihoods)
    plt.show()
    
    #from gaussian -> f^-1 -> data distribution
    #normal = tf.truncated_normal((2,784), stddev=0.1)
    normal = tf.concat((tf.zeros((1,784)), tf.truncated_normal((2,784))),0)
    original_dist = backward_pass(normal, layer_num, getMask((normal.eval().shape[0],784)), False)
    #plt.imshow(np.reshape(original_dist.eval(), (28, 28)), cmap=plt.cm.gray)
    drawMNISTs(original_dist.eval().T)
    #sess.close()
    #del sess 
    
    