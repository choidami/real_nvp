import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name = name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name = name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
  
def gaussianDistribution(h):
    return tf.multiply(-0.5, tf.add(tf.square(h), np.log(2*np.pi)))
  
def logLikelihood(params):
    h1, h2, s = feed_forward(params,iter)
    #reg = -L2_reg * l2_norm(params)
    return -np.sum(gaussianDistribution(h1) + gaussianDistribution(h2) + s)# - reg
    
def getMask(shape):
    dim = int(np.sqrt(shape[1]))
    return np.reshape([1 if (i+j)%2 else 0 for i in xrange(dim) for j in xrange(dim)]*shape[0], shape)

def convnet(inputs):
    W_conv1 = tf.get_variable("W_conv1")
    b_conv1 = tf.get_variable("b_conv1")
    W_conv2 = tf.get_variable("W_conv2")
    b_conv2 = tf.get_variable("b_conv2")
    W_fc = tf.get_variable("W_fc")
    b_fc = tf.get_variable("b_fc")
    
    h_conv1 = tf.nn.relu(conv2d(inputs, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) 
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) 
    return tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)

def forward_layer(x_input, variable_scope, mask):
    with tf.variable_scope(variable_scope):
        with tf.variable_scope("s"):
            scale = convnet(x_input*mask)
        with tf.variable_scope("t"):
            translation = convnet(x_input*mask)
    return mask*x_input + (1-mask)*(x_input*np.exp(scale) + translation)

def backwards_layer(y_input, variable_scope, mask):
    with tf.variable_scope(variable_scope):
        with tf.variable_scope("s"):
            scale = convnet(y_input*mask)
        with tf.variable_scope("t"):
            translation = convnet(y_input*mask)
    return ((1-mask)*y_input - translation)*np.exp(-scale)
  
  
def feed_forward(inputs, mask):
    s_layer1 = neural_net_predict(params[0], x1)
    y1_layer1, y2_layer1 = x1, x2*np.exp(s_layer1) + neural_net_predict(params[1], x1)
    
    s_layer2 = neural_net_predict(params[2], y2_layer1)
    y1_layer2, y2_layer2 = y2_layer1, y1_layer1*np.exp(s_layer2) + neural_net_predict(params[3], y2_layer1)
    
    s_layer3 = neural_net_predict(params[4], y1_layer2)
    y1_layer3, y2_layer3 = y1_layer2, y2_layer2*np.exp(s_layer3) + neural_net_predict(params[5], y1_layer2)
    #y1_layer4, y2_layer4 = y2_layer3, y1_layer3*np.exp(np.tanh(y2_layer3*params[12] + params[13])) + np.tanh(y2_layer3*params[14] + params[15])
    #y1_layer5, y2_layer5 = y1_layer4, y2_layer4*np.exp(np.maximum(0,y1_layer4*params[16] + params[17])) + np.tanh(y1_layer4*params[18] + params[19])
    
    s = s_layer1 + s_layer2 + s_layer3

    return y1_layer3, y2_layer3, s

def drawMNISTs(digits): # plots MNIST from a [784, num_digits] array.
  plt.figure(1)
  plt.clf()
  for i in xrange(digit.shape[1]):
    plt.subplot(1, digit.shape[1], i+1)
    plt.imshow(digit[:, i].reshape(28, 28).T, cmap=plt.cm.gray)
  plt.draw()
  raw_input('Press Enter.')
  
def drawMNIST(digit, title): # plots MNIST from a [784, num_digits] array.
  #plt.figure(1)
  plt.clf()
  plt.title(title)
  plt.imshow(digit.reshape(28, 28).T, cmap=plt.cm.gray)
  plt.title(title)
  plt.draw()
  raw_input('Press Enter.')


if __name__ == '__main__':
    layer_num = 3
    batch_size = 50
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.InteractiveSession(config=config)    
    
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    mask = getMask(batch_size, 784)
    
    #x = tf.placeholder(tf.float32, shape=[None, 784]) # input data
    #mask = tf.placeholder(tf.float32, shape=[None, 784])
    #mask_np = np.reshape(np.array([i%2 for i in xrange(784)]*batch_size), [batch_size, 784])
                      
    # weights for convnets [s1, t1, s2, t2, s3, t3]
    layer_names = ["layer" + str(i+1) + "/s/" for i in xrange(layer_num)] + ["layer" + str(i+1) + "/t/" for i in xrange(layer_num)]
    W_conv1 = [weight_variable([5, 5, 1, 32], layer_names[i] + "W_conv1") for i in layer_names]
    b_conv1 = [bias_variable([32], layer_names[i] + "b_conv1") for i in layer_names]
    W_conv2 = [weight_variable([5, 5, 32, 64], layer_names[i] + "W_conv2_") for i in layer_names]
    b_conv2 = [bias_variable([64], layer_names[i] + "b_conv2_") for i in layer_names]
    W_fc_s1 = [weight_variable([7 * 7 * 64, 784], layer_names[i] + "w_fc_") for i in layer_names]
    b_fc_s1 = [bias_variable([784], layer_names[i] + "b_fc_") for i in layer_names]
    
    
    W_conv1_s1, W_conv1_t1, W_conv1_s2, W_conv1_t2, W_conv1_s3, W_conv1_t3 = [weight_variable([5, 5, 1, 32], "W_conv1_" + str(i)) for i in xrange(layer_num*2)]
    b_conv1_s1, b_conv1_t1, b_conv1_s2, b_conv1_t2, b_conv1_s3, b_conv1_t3 = [bias_variable([32], "b_conv1_" + str(i)) for i in xrange(layer_num*2)]
    W_conv2_s1, W_conv2_t1, W_conv2_s2, W_conv2_t2, W_conv2_s3, W_conv2_t3 = [weight_variable([5, 5, 32, 64], "W_conv2_" + str(i)) for i in xrange(layer_num*2)]
    b_conv2_s1, b_conv2_t1, b_conv2_s2, b_conv2_t2, b_conv2_s3, b_conv2_t3 = [bias_variable([64], "b_conv2_" + str(i)) for i in xrange(layer_num*2)]
    W_fc_s1, W_fc_t1, W_fc_s2, W_fc_t2, W_fc_s3, W_fc_t3 = [weight_variable([7 * 7 * 64, 784], "w_fc_" + str(i)) for i in xrange(layer_num*2)]
    b_fc_s1, b_fc_t1, b_fc_s2, b_fc_t2, b_fc_s3, b_fc_t3, = [bias_variable([784], "b_fc_" + str(i)) for i in xrange(layer_num*2)]
     
    ######################## layer 1 ########################
    x_image = tf.reshape(tf.multiply(x, mask), [-1,28,28,1])
    
    # s layer1
    h_conv1_s1 = tf.nn.relu(conv2d(x_image, W_conv1_s1) + b_conv1_s1)
    h_pool1_s1 = max_pool_2x2(h_conv1_s1)
    h_conv2_s1 = tf.nn.relu(conv2d(h_pool1_s1, W_conv2_s1) + b_conv2_s1) 
    h_pool2_s1 = max_pool_2x2(h_conv2_s1)
    h_pool2_flat_s1 = tf.reshape(h_pool2_s1, [-1, 7*7*64]) 
    h_fc_s1 = tf.nn.relu(tf.matmul(h_pool2_flat_s1, W_fc_s1) + b_fc_s1) 
    
    # t layer1
    h_conv1_t1 = tf.nn.relu(conv2d(x_image, W_conv1_t1) + b_conv1_t1)
    h_pool1_t1 = max_pool_2x2(h_conv1_t1)
    h_conv2_t1 = tf.nn.relu(conv2d(h_pool1_t1, W_conv2_t1) + b_conv2_t1) 
    h_pool2_t1 = max_pool_2x2(h_conv2_t1)
    h_pool2_flat_t1 = tf.reshape(h_pool2_t1, [-1, 7*7*64]) 
    h_fc_t1 = tf.nn.relu(tf.matmul(h_pool2_flat_t1, W_fc_t1) + b_fc_t1) 
    
    y_layer1_output = tf.multiply(mask, x) + tf.multiply(tf.subtract(1.0,mask), tf.add(tf.multiply(x, tf.exp(h_fc_s1)), h_fc_t1))
    
    ######################## layer 2 ########################
    #x2 = tf.placeholder(tf.float32, shape=[batch_size, 784])
    y_layer2_input = tf.reshape(tf.multiply(y_layer1_output, mask), [-1,28,28,1])
    
    # s layer2
    h_conv1_s2 = tf.nn.relu(conv2d(y_layer2_input, W_conv1_s2) + b_conv1_s2)
    h_pool1_s2 = max_pool_2x2(h_conv1_s2)
    h_conv2_s2 = tf.nn.relu(conv2d(h_pool1_s2, W_conv2_s2) + b_conv2_s2) 
    h_pool2_s2 = max_pool_2x2(h_conv2_s2)
    h_pool2_flat_s2 = tf.reshape(h_pool2_s2, [-1, 7*7*64]) 
    h_fc_s2 = tf.nn.relu(tf.matmul(h_pool2_flat_s2, W_fc_s2) + b_fc_s2) 
    
    # t layer2
    h_conv1_t2 = tf.nn.relu(conv2d(y_layer2_input, W_conv1_t2) + b_conv1_t2)
    h_pool1_t2 = max_pool_2x2(h_conv1_t2)
    h_conv2_t2 = tf.nn.relu(conv2d(h_pool1_t2, W_conv2_t2) + b_conv2_t2) 
    h_pool2_t2 = max_pool_2x2(h_conv2_t2)
    h_pool2_flat_t2 = tf.reshape(h_pool2_t2, [-1, 7*7*64]) 
    h_fc_t2 = tf.nn.relu(tf.matmul(h_pool2_flat_t2, W_fc_t2) + b_fc_t2)
    
    y_layer2_output = tf.multiply(tf.subtract(1.0,mask), y_layer1_output) + tf.multiply(mask, tf.add(tf.multiply(y_layer1_output, tf.exp(h_fc_s2)), h_fc_t2))
    
    ######################## layer 3 ########################
    #x3 = tf.placeholder(tf.float32, shape=[batch_size, 784])
    y_layer3_input = tf.reshape(tf.multiply(y_layer2_output, mask), [-1,28,28,1])
    
    # s layer3
    h_conv1_s3 = tf.nn.relu(conv2d(y_layer3_input, W_conv1_s3) + b_conv1_s3)
    h_pool1_s3 = max_pool_2x2(h_conv1_s3)
    h_conv2_s3 = tf.nn.relu(conv2d(h_pool1_s3, W_conv2_s3) + b_conv2_s3) 
    h_pool2_s3 = max_pool_2x2(h_conv2_s3)
    h_pool2_flat_s3 = tf.reshape(h_pool2_s3, [-1, 7*7*64]) 
    h_fc_s3 = tf.nn.relu(tf.matmul(h_pool2_flat_s3, W_fc_s3) + b_fc_s3) 
    
    # t layer3
    h_conv1_t3 = tf.nn.relu(conv2d(y_layer3_input, W_conv1_t3) + b_conv1_t3)
    h_pool1_t3 = max_pool_2x2(h_conv1_t3)
    h_conv2_t3 = tf.nn.relu(conv2d(h_pool1_t3, W_conv2_t3) + b_conv2_t3) 
    h_pool2_t3 = max_pool_2x2(h_conv2_t3)
    h_pool2_flat_t3 = tf.reshape(h_pool2_t3, [-1, 7*7*64]) 
    h_fc_t3 = tf.nn.relu(tf.matmul(h_pool2_flat_t3, W_fc_t3) + b_fc_t3)
    
    y_layer3_output = tf.multiply(mask, y_layer2_output) + tf.multiply(tf.subtract(1.0,mask), tf.add(tf.multiply(y_layer2_output, tf.exp(h_fc_s3)), h_fc_t3))
    
    ######################## Inverse Layers ########################
    h = np.zeros([batch_size, 784])
    y_layer2 = tf.multiply(mask, h) + tf.multiply(tf.subtract(1.0,mask), tf.multiply(tf.subtract(tf.multiply(tf.subtract(1.0,mask),h), h_fc_t3.eval(feed_dict={y_layer3_input:np.reshape(mask_np*h, [-1,28,28,1])})), tf.exp(-h_fc_s3.eval(feed_dict={y_layer3_input:np.reshape(mask_np*h, [-1,28,28,1])}))))
    
    # Calculate log-likelihood
    s = h_fc_s1 + h_fc_s2 + h_fc_s3
    log_likelihood = -tf.reduce_mean(gaussianDistribution(y_layer3_output))# - tf.reduce_mean(s)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(log_likelihood)
    
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
      batch = mnist.train.next_batch(batch_size)
      if i%100 == 0:
          #drawMNIST(batch[0][0], str(batch[1][0]))
          #h = tf.zeros([batch_size, 784], tf.float32)
          #y_layer2 = tf.multiply(mask, h) + tf.multiply(tf.subtract(1.0,mask), tf.add(tf.multiply(h, tf.exp(h_fc_s2)), h_fc_t2))
          print("step %d, log-likelihood %g"%(i, log_likelihood.eval(feed_dict={x: batch[0], mask: mask_np})))
          #print(W_conv1[0].eval())
          #print("step %d, log-likelihood %g"%(i, log_likelihood.eval(feed_dict={x: batch[0]})))
        
      train_step.run(feed_dict={x: batch[0], mask:mask_np})
    
    mask_final = np.reshape(np.array([float(i%2) for i in xrange(784)]*len(mnist.test.images)), [len(mnist.test.images), 784])
    print("log-likelihood %g"%log_likelihood.eval(feed_dict={x: mnist.test.images, mask:mask_final}))
    #sess.close()
    #del sess 
      #y1_layer2, y2_layer2 = h1, (h2 - neural_net_predict(params[5], h1))*np.exp(-neural_net_predict(params[4], h1))
    #y1_layer1, y2_layer1 = (y2_layer2 - neural_net_predict(params[3], y1_layer2))*np.exp(-neural_net_predict(params[2], y1_layer2)), y1_layer2
    #x1, x2 = y1_layer1, (y2_layer1 - neural_net_predict(params[1], y1_layer1))*np.exp(-neural_net_predict(params[0], y1_layer1))
    
    