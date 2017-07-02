import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import real_nvp.nn as real_nvp_nn
from real_nvp.model import model_spec as real_nvp_model_spec
from real_nvp.model import inv_model_spec as real_nvp_inv_model_spec
import util
import plotting

       
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
    # parameters
    batch_size = 1
    sample_size = 5
    num_epoch = 200
    learning_rate = 0.001
    load_params = False # load_params=False trains from scratch
    save_interval = 100
    
    # Load Data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    model_spec = real_nvp_model_spec
    inv_model_spec = real_nvp_inv_model_spec
    nn = real_nvp_nn
    
    # create the model
    model = tf.make_template('model', model_spec)
    inv_model = tf.make_template('model', inv_model_spec, unique_name_='model')
    
    x_init = tf.placeholder(tf.float32, shape=(batch_size,28,28,1))
    # run once for data dependent initialization of parameters
    gen_par = model(x_init)
    
    # sample from the model
    x_sample = tf.placeholder(tf.float32, shape=(sample_size,28,28,1))
    new_x_gen = inv_model(x_sample)
    def sample_from_model(sess):
        global sample_size
        x_gen = np.random.normal(0.0, 0.1, (sample_size, 28, 28, 1))
        new_x_gen_np = sess.run(new_x_gen, {x_sample:x_gen})
        return new_x_gen_np
    
    # get loss gradients over GPU
    all_params = tf.trainable_variables()
    with tf.device('/gpu:0'):
        # train
        x_input = tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1))
        gen_par, jacs = model(x_input)
        loss_gen = nn.loss(gen_par, jacs)
        # gradients
        grad = tf.gradients(loss_gen, all_params)
        # test
        gen_par, jacs = model(x_input)
        loss_gen_test = nn.loss(gen_par, jacs)
        
        # training op
        tf_lr = tf.placeholder(tf.float32, shape=[])
        optimizer = nn.adam_updates(all_params, grad, lr=tf_lr, mom1=0.99, mom2=0.9995)
        
    # init & save
    initializer = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    # input is scaled from uint8 [0, 255] to float in range[-1, 1]
    def prepro(x):
        return np.cast[np.float32]((x - 127.5) / 127.5)
    
    # //////////// perform training //////////////
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.InteractiveSession(config=config) 
    
    x = mnist.train.next_batch(batch_size)
    batch = np.reshape(x[0], (batch_size, 28, 28, 1)) 
    print('starting training')
    log_likelihoods = []
    lr = learning_rate
        
    for epoch in range(num_epoch):
            
        #x = mnist.train.next_batch(batch_size)
        #batch = np.reshape(x[0], (batch_size, 28, 28, 1))
        # init
        if epoch == 0:
                
            print('initializing the model...')
            sess.run(initializer,{x_init: batch})
                
            # TODO: load params if load_params==True
            if load_params:
                #ckpt_file = 'final_mnist_generator.ckpt'
                ckpt_file = "test11.ckpt"
                print('restoring parameters from', ckpt_file)
                saver.restore(sess, ckpt_file)
                
                
            #util.show_all_variables()
                
        # train for one epoch
        #print("Training (%d/%d) started" %(epoch, num_epoch))
        feed_dict = {x_input:batch, tf_lr:learning_rate}
        ll,_ = sess.run([loss_gen, optimizer], feed_dict)
            
        #drawMNISTs(batch)
            
        if epoch % save_interval == 0:
                # log progress to console
            print("Iteration %d, train log likelihood %.4f" %(epoch, ll))
            log_likelihoods.append(ll)
            #sys.stdout.flush()
        
    #plot log-likelihoods
    plt.clf()
    x_axis = np.linspace(0,num_epoch, num_epoch/save_interval)    
    plt.plot(x_axis, log_likelihoods)
    plt.show()      
        
    print("Generating samples...")
    #generate samples from the model
    sampled_x = sample_from_model(sess)
    drawMNISTs(sampled_x)
        
    # save params   
    saver.save(sess, 'test_with_kernel5x5.ckpt')
    #saver.save(sess, 'final_mnist_generator3.ckpt')
    
    
    
#    slim = tf.contrib.slim
#    
#    layer_num = 6
#    batch_size = 50
#    num_epoch = 500
#    
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth=True
#    sess = tf.InteractiveSession(config=config)    
#    
#    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#    
#    x = tf.placeholder(tf.float32, shape=[None, 784]) #input images
#    y_ = tf.placeholder(tf.float32, shape=[None, 10]) #labels
#    is_training = tf.placeholder(tf.bool)
#    mask = tf.placeholder(tf.float32, shape=[None, 784])
#    
#    # Calculate log-likelihood for redl-NVP
#    h, s = forward_pass(x, layer_num, mask, is_training)
#    log_likelihood = -tf.reduce_sum(gaussianDistribution(h)+s)
#    
#    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#    with tf.control_dependencies(update_ops):
#        train_step_generator = tf.train.AdamOptimizer(
#                learning_rate=0.001,
#                beta1=1. - 1e-1,
#                beta2=1. - 1e-3,
#                epsilon=1e-08).minimize(log_likelihood)
#        
#    # Loss function for Z Classifier 
#    output_Z_classifier = z_classifier(x, is_training)
#    cross_entropy_Z = tf.reduce_mean(
#            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output_Z_classifier))
#    train_step_Z_classifier = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_Z) 
#    correct_prediction_Z = tf.equal(tf.argmax(output_Z_classifier,1), tf.argmax(y_,1))
#    accuracy_Z = tf.reduce_mean(tf.cast(correct_prediction_Z, tf.float32))
#        
#    
#    
#    sess.run(tf.global_variables_initializer())
#    
#    # Create a saver object to save all the variables
#    saver = tf.train.Saver()
#    
#    log_likelihoods = []
#    batch_mask = getMask((batch_size, 784))
#    for i in range(num_epoch):
#        batch = mnist.train.next_batch(batch_size)
#        if i%100 == 0:
#            loglikelihood = log_likelihood.eval(feed_dict={x: batch[0], mask: batch_mask, is_training:False})
#            print("step %d, log-likelihood %g"%(i, loglikelihood))
#            log_likelihoods.append(loglikelihood)
#        train_step_generator.run(feed_dict={x: batch[0], mask: getMask((batch_size, 784)), is_training:True})
#        
##    log_likelihoods = []
##    
##    for i in range(num_epoch):
##      batch = mnist.train.next_batch(batch_size)
##      if i%100 == 0:
##          loglikelihood = log_likelihood.eval(feed_dict={x: batch[0], mask: getMask((batch_size, 784)), is_training:False})
##          log_likelihoods.append(loglikelihood)
##          print("step %d, log-likelihood %g"%(i, loglikelihood))
##          #output = h.eval(feed_dict={x: batch[0], mask: getMask((batch_size, 784))})[0]
##          #plt.figure(i)
##          #plt.clf()
##          #plt.imshow(np.reshape(output, (28, 28)), cmap=plt.cm.gray)
##          #plt.draw()
##          #if i%10000 == 0 and i is not 0:
##          #    saver.save(sess, 'my_test_model_' + str(i))
##      train_step_generator.run(feed_dict={x: batch[0], mask: getMask((batch_size, 784)), is_training:True})
#    #saver.save(sess, 'my_last_model')  
#
#    
#    #get test-set log-likelihood
#    #mask_test = getMask((len(mnist.test.images), 784))
#    #print("log-likelihood %g"%log_likelihood.eval(feed_dict={x: mnist.test.images, mask: mask_test}))
#    
#    #plot log-likelihoods
#    plt.clf()
#    x_axis = np.linspace(0,num_epoch, num_epoch/100)    
#    plt.plot(x_axis, log_likelihoods)
#    plt.show()
#    
#    #from gaussian -> f^-1 -> data distribution
#    normal = tf.truncated_normal((2,784))
#    #normal = tf.constant(np.random.randn(1,784), tf.float32)
#    #normal = tf.truncated_normal((2,784), stddev=0.1)
#    original_dist = backward_pass(normal, layer_num, getMask((normal.eval().shape[0],784)), False, reuse=True)
#    #plt.imshow(np.reshape(original_dist.eval(), (28, 28)), cmap=plt.cm.gray)
#    drawMNISTs(original_dist.eval())    
#    
#    
#    # Train the z-classifier (computes p(class|z))
#    for i in range(20000):
#        batch = mnist.train.next_batch(batch_size)
#        z,_ = forward_pass(tf.cast(batch[0], tf.float32), layer_num, batch_mask, False, reuse=True)
#        if i%1000 == 0:
#            train_accuracy = accuracy_Z.eval(feed_dict={x:z.eval(), y_: batch[1], is_training:False})
#            print("step %d, training accuracy %g"%(i, train_accuracy))        
#        train_step_Z_classifier.run(feed_dict={x: z.eval(), y_: batch[1], is_training:True})
#    
#    test_set,_ = forward_pass(mnist.test.images, layer_num, batch_mask, False, reuse=True)
#    print("test accuracy %g"%accuracy_Z.eval(feed_dict={
#        x: test_set.eval(), y_: mnist.test.labels, is_training:False}))
#    
#    
#    
#    #sess.close()
#    #del sess 
    
    
    
    
    
    