import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape, name):
  return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(0, 0.1))

def bias_variable(shape, name):
  return tf.get_variable(name, shape, initializer = tf.constant_initializer(0.1))

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
  
def gaussianDistribution(h):
    return -0.5*(h**2 + np.log(2*np.pi))
    
def getMask(shape):
    dim = int(np.sqrt(shape[1]))
    return np.reshape([1 if (i+j)%2 else 0 for i in xrange(dim) for j in xrange(dim)]*shape[0], shape)

def convnet(inputs):
    tf.get_variable_scope().reuse_variables()
    inputs = tf.reshape(inputs, [-1,28,28,1])
    W_conv1 = tf.get_variable("W_conv1")
    b_conv1 = tf.get_variable("b_conv1")
    W_conv2 = tf.get_variable("W_conv2")
    b_conv2 = tf.get_variable("b_conv2")
    W_fc1 = tf.get_variable("W_fc1")
    b_fc1 = tf.get_variable("b_fc1")
    W_fc2 = tf.get_variable("W_fc2")
    b_fc2 = tf.get_variable("b_fc2")
    
    h_conv1 = conv2d(inputs, W_conv1) + b_conv1
    h_pool1 = max_pool_2x2(tf.nn.relu(h_conv1))
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) 
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) 
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    return tf.nn.tanh(y_conv)

def forward_layer(x_input, variable_scope, mask):
    with tf.variable_scope(variable_scope):
        with tf.variable_scope("s"):
            scale = convnet(x_input*mask)
        with tf.variable_scope("t"):
            translation = convnet(x_input*mask)
    return (mask*x_input + (1-mask)*(x_input*tf.exp(scale) + translation), scale)

def backwards_layer(y_input, variable_scope, mask):
    with tf.variable_scope(variable_scope):
        with tf.variable_scope("s"):
            scale = convnet(y_input*mask)
        with tf.variable_scope("t"):
            translation = convnet(y_input*mask)
    return ((1-mask)*y_input - translation)*tf.exp(-scale)

def forward_pass(x, layer_num, mask):
    scale = 0
    for i in xrange(layer_num):
        y = x
        if i%2:
            y, s = forward_layer(y, "layer"+str(i+1),(1-mask))
        else:
            y, s = forward_layer(y, "layer"+str(i+1) , mask)
        scale += s
    return y, scale
    
def backward_pass(y, layer_num, mask):
    if not layer_num%2: # if layer_num is even, then revert the mask. 
        mask = (1-mask)
        
    for i in xrange(layer_num):
        x = y
        if i%2:
            x = backwards_layer(x, "layer"+str(i+1),(1-mask))
        else:
            x = backwards_layer(x, "layer"+str(i+1) , mask)
    return x
  
def drawMNISTs(digits): # plots MNIST from a [784, num_digits] array.
  plt.figure(1)
  plt.clf()
  for i in xrange(digit.shape[1]):
    plt.subplot(1, digit.shape[1], i+1)
    plt.imshow(digit[:, i].reshape(28, 28).T, cmap=plt.cm.gray)
  plt.draw()
  raw_input('Press Enter.')
  
def drawMNIST(digit, title): # plots MNIST from a [784, num_digits] array.
  plt.figure(1)
  plt.clf()
  plt.title(title)
  plt.imshow(np.reshape(digit, (28, 28)), cmap=plt.cm.gray)
  plt.title(title)
  plt.draw()
  raw_input('Press Enter.')


if __name__ == '__main__':
    layer_num = 5
    batch_size = 50
    num_epoch = 10000
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.InteractiveSession(config=config)    
    
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
                      
    # weights for convnets: layer<layer number>/<s or t>/<variable name>
    layer_names = ["layer" + str(i+1) + "/s/" for i in xrange(layer_num)] + ["layer" + str(i+1) + "/t/" for i in xrange(layer_num)]
    W_conv1 = [weight_variable([5, 5, 1, 32], i + "W_conv1") for i in layer_names]
    b_conv1 = [bias_variable([32], i + "b_conv1") for i in layer_names]
    W_conv2 = [weight_variable([5, 5, 32, 64], i + "W_conv2") for i in layer_names]
    b_conv2 = [bias_variable([64], i + "b_conv2") for i in layer_names]
    W_fc1 = [weight_variable([7 * 7 * 64, 1024], i + "W_fc1") for i in layer_names]
    b_fc1 = [bias_variable([1024], i + "b_fc1") for i in layer_names]
    W_fc2 = [weight_variable([1024, 784], i + "W_fc2") for i in layer_names]
    b_fc2 = [bias_variable([784], i + "b_fc2") for i in layer_names]
    
    x = tf.placeholder(tf.float32, shape=[None, 784])    
    mask = tf.placeholder(tf.float32, shape=[None, 784])
    h, s = forward_pass(x, layer_num, mask)
    
    # Calculate log-likelihood
    log_likelihood = -tf.reduce_mean(gaussianDistribution(h)+s)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the train_step
        train_step = tf.train.AdamOptimizer(1e-4).minimize(log_likelihood)
    
    log_likelihoods = []
    
    sess.run(tf.global_variables_initializer())
    for i in range(num_epoch):
      batch = mnist.train.next_batch(batch_size)
      if i%100 == 0:
          loglikelihood = log_likelihood.eval(feed_dict={x: batch[0], mask: getMask((batch_size, 784))})
          log_likelihoods.append(loglikelihood)
          print("step %d, log-likelihood %g"%(i, loglikelihood))
          #output = h.eval(feed_dict={x: batch[0], mask: getMask((batch_size, 784))})[0]
          #plt.figure(i)
          #plt.clf()
          #plt.imshow(np.reshape(output, (28, 28)), cmap=plt.cm.gray)
          #plt.draw()
        
      train_step.run(feed_dict={x: batch[0], mask: getMask((batch_size, 784))})
    
    #get test-set log-likelihood
    #mask_test = getMask((len(mnist.test.images), 784))
    #print("log-likelihood %g"%log_likelihood.eval(feed_dict={x: mnist.test.images, mask: mask_test}))
    
    #plot log-likelihoods
    plt.clf()
    x_axis = np.linspace(0,num_epoch, num_epoch/100)    
    plt.plot(x_axis, log_likelihoods)
    plt.show()
    
    #from gaussian -> f^-1 -> data distribution
    normal = tf.truncated_normal((1,784), stddev=0.1)
    original_dist = backward_pass(normal, layer_num, getMask((1,784)))
    #plt.imshow(np.reshape(original_dist.eval(), (28, 28)), cmap=plt.cm.gray)
    drawMNIST(original_dist.eval(), "I don't know what this is")
    #sess.close()
    #del sess 
    
    