import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def get_normalized_weights(name, weights_shape):
    weights = tf.get_variable(name, weights_shape, tf.float32,
                              tf.contrib.layers.xavier_initializer())
    norm = tf.sqrt(tf.reduce_sum(tf.square(weights)))
    return weights/norm
  
def gaussianDistribution(h):
    return -0.5*(h**2 + np.log(2*np.pi))
    
def getMask(shape):
    dim = int(np.sqrt(shape[1]))
    return np.reshape([1 if (i+j)%2 else 0 for i in xrange(dim) for j in xrange(dim)]*shape[0], shape)

def batch_norm(x):
    mu = tf.reduce_mean(x)
    sig2 = tf.reduce_mean(tf.square(x-mu))
    x = (x-mu)/tf.sqrt(sig2 + 1.0e-6)
    return x

def mnist_classifier(inputs, is_training, scope = "mnist_classifier", reuse=False):
    inputs = tf.reshape(inputs, [-1,28,28,1])
    with tf.variable_scope(scope, "mnist_classifier", [inputs], reuse=reuse):
        with slim.arg_scope([slim.conv2d],
                            kernel_size=[5,5],
                            stride=1,
                            activation_fn=tf.nn.relu,
                            normalizer_params={'is_training': is_training}
                            ):
            h_conv1 = slim.conv2d(inputs, 32)
            h_pool1 = slim.max_pool2d(h_conv1, 2, [2,2])
    
            h_conv2 = slim.conv2d(h_pool1, 64)
            h_pool2 = slim.max_pool2d(h_conv2, 2, [2,2])
            
            h_pool2_flat = slim.flatten(h_pool2)
            
            h_fc1 = slim.fully_connected(h_pool2_flat, 1024, activation_fn = tf.nn.relu)
                        
            y_conv = slim.fully_connected(h_fc1, 10, activation_fn=None)
            
            return y_conv

def s_or_t(inputs, is_training, scope="s_or_t"):
    with tf.variable_scope(scope, "s_or_t", [inputs]) as var_scope:
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            normalizer_params={'is_training': is_training}
                            #weights_regularizer=slim.l2_regularizer(0.01)
                            ):
            net = inputs
            for i in xrange(5):
                net = slim.fully_connected(net, 1000, scope='fc'+str(i+1))
                #Add a dropout layer to prevent over-fittting
                #net = slim.dropout(net, 0.8, is_training=is_training)
            if "/s/" in var_scope.name:
                predictions = slim.fully_connected(net, 784, activation_fn=tf.tanh)
                scale_factor = get_normalized_weights("weights_tanh_scale_s",[1])
                #print(scale_factor.eval())
            else:
                predictions = slim.fully_connected(net, 784, activation_fn=None)
                scale_factor = 1#get_normalized_weights("weights_tanh_scale_t",[1])
            return predictions*scale_factor
        
def z_classifier(inputs, is_training, scope="z_classifier", reuse=False):
    with tf.variable_scope(scope, "z_classifier", [inputs], reuse=reuse):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            normalizer_params={'is_training': is_training}
                            #weights_regularizer=slim.l2_regularizer(0.01)
                            ):
            net = inputs
            for i in xrange(5):
                net = slim.fully_connected(net, 1000, scope='fc'+str(i+1))
                #Add a dropout layer to prevent over-fittting
                #net = slim.dropout(net, 0.8, is_training=is_training)
            predictions = slim.fully_connected(net, 10, activation_fn=None)
            return predictions  

def forward_layer(x_input, variable_scope, mask, is_training, reuse):
    with tf.variable_scope(variable_scope, reuse=reuse):
        with tf.variable_scope("s"):
            scale = s_or_t(x_input*mask, is_training)
        with tf.variable_scope("t"):
            translation = s_or_t(x_input*mask, is_training)
    return (mask*x_input + (1-mask)*(x_input*tf.exp(scale) + translation), scale)

def backwards_layer(y_input, variable_scope, mask, is_training, reuse):
    with tf.variable_scope(variable_scope, reuse=reuse):
        with tf.variable_scope("s"):
            scale = s_or_t(y_input*mask, is_training)
        with tf.variable_scope("t"):
            translation = s_or_t(y_input*mask, is_training)
    return mask*y_input + (1-mask)*(y_input - translation)*tf.exp(-scale)

def forward_pass(x, layer_num, mask, is_training, reuse=False):
    scale = 0
    y = x
    for i in xrange(layer_num):
        if i%2:
            y, s = forward_layer(y, "layer"+str(i+1),(1-mask), is_training, reuse)
        else:
            y, s = forward_layer(y, "layer"+str(i+1) , mask, is_training, reuse)
        scale += s
    return y, scale
    
def backward_pass(y, layer_num, mask, is_training, reuse=False):
    x = y    
    for i in xrange(layer_num-1, -1, -1):
        if i%2:
            x = backwards_layer(x, "layer"+str(i+1),(1-mask), is_training, reuse)
        else:
            x = backwards_layer(x, "layer"+str(i+1) , mask, is_training, reuse)
    return x
  
def drawMNISTs(digits): # plots MNIST from a [784, num_digits] array.
  for i in range(digits.shape[0]):
    plt.figure()
    plt.imshow(digits[i, :].reshape(28, 28), cmap=plt.cm.gray)
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
    slim = tf.contrib.slim
    
    layer_num = 6
    batch_size = 50
    num_epoch = 500
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.InteractiveSession(config=config)    
    
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    x = tf.placeholder(tf.float32, shape=[None, 784]) #input images
    y_ = tf.placeholder(tf.float32, shape=[None, 10]) #labels
    is_training = tf.placeholder(tf.bool)
    mask = tf.placeholder(tf.float32, shape=[None, 784])
    
    # Calculate log-likelihood for redl-NVP
    h, s = forward_pass(x, layer_num, mask, is_training)
    log_likelihood = -tf.reduce_sum(gaussianDistribution(h)+s)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step_generator = tf.train.AdamOptimizer(
                learning_rate=0.001,
                beta1=1. - 1e-1,
                beta2=1. - 1e-3,
                epsilon=1e-08).minimize(log_likelihood)
        
    # Loss function for Z Classifier 
    output_Z_classifier = z_classifier(x, is_training)
    cross_entropy_Z = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output_Z_classifier))
    train_step_Z_classifier = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_Z) 
    correct_prediction_Z = tf.equal(tf.argmax(output_Z_classifier,1), tf.argmax(y_,1))
    accuracy_Z = tf.reduce_mean(tf.cast(correct_prediction_Z, tf.float32))
        
    
    
    sess.run(tf.global_variables_initializer())
    
    # Create a saver object to save all the variables
    saver = tf.train.Saver()
    
    log_likelihoods = []
    batch_mask = getMask((batch_size, 784))
    for i in range(num_epoch):
        batch = mnist.train.next_batch(batch_size)
        if i%100 == 0:
            loglikelihood = log_likelihood.eval(feed_dict={x: batch[0], mask: batch_mask, is_training:False})
            print("step %d, log-likelihood %g"%(i, loglikelihood))
            log_likelihoods.append(loglikelihood)
        train_step_generator.run(feed_dict={x: batch[0], mask: getMask((batch_size, 784)), is_training:True})
        
#    log_likelihoods = []
#    
#    for i in range(num_epoch):
#      batch = mnist.train.next_batch(batch_size)
#      if i%100 == 0:
#          loglikelihood = log_likelihood.eval(feed_dict={x: batch[0], mask: getMask((batch_size, 784)), is_training:False})
#          log_likelihoods.append(loglikelihood)
#          print("step %d, log-likelihood %g"%(i, loglikelihood))
#          #output = h.eval(feed_dict={x: batch[0], mask: getMask((batch_size, 784))})[0]
#          #plt.figure(i)
#          #plt.clf()
#          #plt.imshow(np.reshape(output, (28, 28)), cmap=plt.cm.gray)
#          #plt.draw()
#          #if i%10000 == 0 and i is not 0:
#          #    saver.save(sess, 'my_test_model_' + str(i))
#      train_step_generator.run(feed_dict={x: batch[0], mask: getMask((batch_size, 784)), is_training:True})
    #saver.save(sess, 'my_last_model')  

    
    #get test-set log-likelihood
    #mask_test = getMask((len(mnist.test.images), 784))
    #print("log-likelihood %g"%log_likelihood.eval(feed_dict={x: mnist.test.images, mask: mask_test}))
    
    #plot log-likelihoods
    plt.clf()
    x_axis = np.linspace(0,num_epoch, num_epoch/100)    
    plt.plot(x_axis, log_likelihoods)
    plt.show()
    
    #from gaussian -> f^-1 -> data distribution
    normal = tf.truncated_normal((2,784))
    #normal = tf.constant(np.random.randn(1,784), tf.float32)
    #normal = tf.truncated_normal((2,784), stddev=0.1)
    original_dist = backward_pass(normal, layer_num, getMask((normal.eval().shape[0],784)), False, reuse=True)
    #plt.imshow(np.reshape(original_dist.eval(), (28, 28)), cmap=plt.cm.gray)
    drawMNISTs(original_dist.eval())    
    
    
    # Train the z-classifier (computes p(class|z))
    for i in range(20000):
        batch = mnist.train.next_batch(batch_size)
        z,_ = forward_pass(tf.cast(batch[0], tf.float32), layer_num, batch_mask, False, reuse=True)
        if i%1000 == 0:
            train_accuracy = accuracy_Z.eval(feed_dict={x:z.eval(), y_: batch[1], is_training:False})
            print("step %d, training accuracy %g"%(i, train_accuracy))        
        train_step_Z_classifier.run(feed_dict={x: z.eval(), y_: batch[1], is_training:True})
    
    test_set,_ = forward_pass(mnist.test.images, layer_num, batch_mask, False, reuse=True)
    print("test accuracy %g"%accuracy_Z.eval(feed_dict={
        x: test_set.eval(), y_: mnist.test.labels, is_training:False}))
    
    
    
    #sess.close()
    #del sess 
    
    
    
    
    
    