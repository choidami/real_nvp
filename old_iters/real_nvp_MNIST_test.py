import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
slim = tf.contrib.slim

def int_shape(x):
    return list(map(int, x.shape))

def get_normalized_weights(name, weights_shape):
    weights = tf.get_variable(name, weights_shape, tf.float32,
                              tf.contrib.layers.xavier_initializer())
    norm = tf.sqrt(tf.reduce_sum(tf.square(weights)))
    return weights/norm
  
def gaussianDistribution(h):
    return -0.5*(h**2 + np.log(2*np.pi))

def loglikelihood(h, s):
    return -tf.reduce_sum(gaussianDistribution(h)+s)
    
def getMask(shape):
    dim = int(np.sqrt(shape[1]))
    return np.reshape([1 if (i+j)%2 else 0 for i in xrange(dim) for j in xrange(dim)]*shape[0], shape)

def z_classifier(inputs, is_training, scope="z_classifier", reuse=False):
    with tf.variable_scope(scope, "z_classifier", [inputs], reuse=reuse):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            normalizer_params={'is_training': is_training}
                            #weights_regularizer=slim.l2_regularizer(0.01)
                            ):
            net = inputs
            for i in xrange(3):
                net = slim.fully_connected(net, 1000, scope='fc'+str(i+1))
                #Add a dropout layer to prevent over-fitting
                net = slim.dropout(net, 0.7, is_training=is_training)
            predictions = slim.fully_connected(net, 10, activation_fn=None)
            return predictions  
        
def s_or_t(inputs, is_training, scope="s_or_t"):
    with tf.variable_scope(scope, "s_or_t", [inputs]) as var_scope:
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            activation_fn=tf.nn.relu,
                            #normalizer_params={'is_training': is_training},
                            weights_regularizer=slim.l2_regularizer(0.00005),
                            ):
            net = inputs
            for i in xrange(5):
                net = slim.fully_connected(net, 1000, scope='fc'+str(i+1))
            if "/s/" in var_scope.name:
                predictions = slim.fully_connected(net, 784, activation_fn=tf.nn.tanh)
                scale_factor = get_normalized_weights("weights_tanh_scale_s",[1])
            else:
                predictions = slim.fully_connected(net, 784, activation_fn=None)
                scale_factor = 1#get_normalized_weights("weights_tanh_scale_t",[1])
            return predictions*scale_factor
        
def forward_layer(x_input, variable_scope, mask, is_training, reuse):
    with tf.variable_scope(variable_scope, reuse=reuse):
        with tf.variable_scope("s"):
            scale = s_or_t(x_input*mask, is_training)
        with tf.variable_scope("t"):
            translation = s_or_t(x_input*mask, is_training)
    return (mask*x_input + (1-mask)*(x_input*tf.check_numerics(tf.exp(scale), "exp has NaN") + translation), scale)

def backwards_layer(y_input, variable_scope, mask, is_training=False, reuse=True):
    with tf.variable_scope(variable_scope, reuse=reuse):
        with tf.variable_scope("s"):
            scale = s_or_t(y_input*mask, is_training)
        with tf.variable_scope("t"):
            translation = s_or_t(y_input*mask, is_training)
    return mask*y_input + (1-mask)*(y_input - translation)*tf.check_numerics(tf.exp(-scale), "exp has NaN")

def forward_pass(x, xs, layer_num, mask, is_training, reuse=False):   
    low, high = 0.0, 1.0
    corruption_level = 1.0
    
    # corrupt data (Tapani Raiko's dequantization)
    y = x - low
    y /= high - low
    y *= 255.0
    y += corruption_level * tf.random_uniform(xs)
    y /= (255 + corruption_level)
    y *= high - low
    y += low
    
#    y = x*0.5 + 0.5
#    y = y*255.0
#    corruption_level = 1.0
#    y = y + corruption_level * tf.random_uniform(xs)
#    y = y/(255.0 + corruption_level)
    
    # model logit instead of the x itself
    alpha = 1e-5
    y = y*(1-alpha) + alpha*0.5
    jac = -tf.log(y) - tf.log(1-y)
    y = tf.log(y) - tf.log(1-y)
    
    #jac = 0
    
    for i in xrange(layer_num):
        if i%2:
            y, s = forward_layer(y, "layer"+str(i+1),(1-mask), is_training, reuse)
        else:
            y, s = forward_layer(y, "layer"+str(i+1) , mask, is_training, reuse)
        jac += s
    return y, jac
    
def backward_pass(y, layer_num):
    ys = int_shape(y)
    mask = getMask(ys)
    x = y    
    for i in xrange(layer_num-1, -1, -1):
        if i%2:
            x = backwards_layer(x, "layer"+str(i+1),(1-mask))
        else:
            x = backwards_layer(x, "layer"+str(i+1) , mask)
    #return x
    return tf.reciprocal(1 + tf.exp(-x))
  
def drawMNISTs(digits): # plots MNIST from a [784, num_digits] array.
  for i in range(digits.shape[0]):
    plt.figure()
    plt.imshow(digits[i, :].reshape(28, 28), cmap=plt.cm.gray)
  raw_input('Press Enter.')

if __name__ == '__main__':
    # parameters
    layer_num = 4
    batch_size = 64
    sample_size = 1
    num_epoch = 30000
    learning_rate = 0.0001
    load_params_gen = True
    cont_train_gen = False
    save_interval = 100
    
    # Load Data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    # create the model
    generator = tf.make_template('generator', forward_pass)
    inv_generator = tf.make_template('generator', backward_pass, unique_name_='generator')
    classifier = tf.make_template('classifier', z_classifier)
    
    # get loglikelihood gradients over GPU
    all_params = tf.trainable_variables()
    with tf.device('/gpu:0'):
        optimizer = tf.train.AdamOptimizer(
                learning_rate=0.0001,
                beta1=1. - 1e-1,
                beta2=1. - 1e-3,
                epsilon=1e-08)
        # Generator 
        x_input = tf.placeholder(tf.float32, shape=[None, 784])
        xs = tf.placeholder(tf.int32, shape=[2])
        is_training = tf.placeholder(tf.bool)
        mask = tf.placeholder(tf.float32, shape=[None, 784])
        reuse = tf.placeholder(tf.bool)
        z, jacs = generator(x_input, xs, layer_num, mask, is_training)
        log_likelihood = loglikelihood(z, jacs)
        gen_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
        train_step_gen = optimizer.minimize(log_likelihood, var_list=gen_train_vars)
        
        # Z classifier
        z_input = tf.placeholder(tf.float32, shape=[None, 784])
        labels = tf.placeholder(tf.float32, shape=[None, 10])
        output_Z_classifier = classifier(z_input, is_training)
        cross_entropy_Z = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output_Z_classifier))
        class_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "classifier")
        train_step_Z_classifier = optimizer.minimize(cross_entropy_Z, var_list=class_train_vars) 
        correct_prediction_Z = tf.equal(tf.argmax(output_Z_classifier,1), tf.argmax(labels,1))
        accuracy_Z = tf.reduce_mean(tf.cast(correct_prediction_Z, tf.float32))
        
    # init & save
    initializer = tf.global_variables_initializer()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.InteractiveSession(config=config)
    sess.run(initializer)
    
    x_sample = tf.placeholder(tf.float32, shape=(sample_size,784))
    new_x_gen = inv_generator(x_sample, layer_num)
    def sample_from_model(sess):
        global sample_size
        x_gen = np.random.normal(0.0, 1.0, (sample_size, 784))
        new_x_gen_np = sess.run(new_x_gen, {x_sample:x_gen})
        return new_x_gen_np

    
    digits = []
    if not load_params_gen:
        # //////////// perform training //////////////
        print('starting training')
        log_likelihoods = []
        batch_mask = getMask((batch_size, 784))   
        
        #x = mnist.train.next_batch(batch_size)
        #batch = x[0]
        for epoch in range(num_epoch):
            batch = mnist.train.next_batch(batch_size)[0]
            if epoch == 0:                
                if cont_train_gen:
                    ckpt_file = 'final_mnist_generator.ckpt'
                    print('restoring parameters from', ckpt_file)
                    saver.restore(sess, ckpt_file)          
            feed_dict={x_input: batch, xs: int_shape(batch), mask: batch_mask, is_training:True}
            ll,_ = sess.run([log_likelihood, train_step_gen], feed_dict)
                
            if epoch % save_interval == 0:
                # log progress to console
                print("Iteration %d, train log likelihood %.4f" %(epoch, ll))
                log_likelihoods.append(ll)
                
                # Generate Samples
                sampled_x = sample_from_model(sess)
                digits.append(sampled_x)
                #sys.stdout.flush()
            
        # save params   
        saver.save(sess, 'mnist_gen.ckpt')
        
        #plot log-likelihoods
        plt.clf()
        x_axis = np.linspace(0,num_epoch, num_epoch/save_interval)    
        #plt.plot(x_axis[10:], log_likelihoods[10:])
        plt.plot(x_axis, log_likelihoods)
        plt.show() 
    else:
        ckpt_file = "mnist_gen.ckpt"
        print('restoring generator from ', ckpt_file)
        saver.restore(sess, ckpt_file) 
        
        
    print("Generating samples...")
    # generate samples from the model
    digits.append(sample_from_model(sess))
    digits = np.vstack(digits)
    #drawMNISTs(digits)
    
     ####################### Training z classifier #######################
    #parameters
    batch_size = 50
    num_epoch = 20000
    learning_rate = 0.0001
    load_params_z = True # load_params=False trains from scratch  
    
    if not load_params_z:
        batch_mask = getMask((batch_size, 784))
        for i in range(num_epoch):
            batch = mnist.train.next_batch(batch_size)
            x_to_z = sess.run(z, feed_dict={x_input:batch[0], xs: int_shape(batch[0]), mask: batch_mask, is_training:False, reuse:True})
            feed_dict = {z_input:x_to_z, labels: batch[1], is_training:False, reuse:True}
            if i%save_interval == 0:
                train_accuracy = sess.run(accuracy_Z, feed_dict)
                print("step %d, training accuracy %g"%(i, train_accuracy))        
            sess.run(train_step_Z_classifier, feed_dict)
        # save params   
        saver.save(sess, 'z_classifier.ckpt') 
    else:
        ckpt_file = "z_classifier.ckpt"
        print('restoring generator from ', ckpt_file)
        saver.restore(sess, ckpt_file) 
        
    
    #test_set,_ = forward_pass(mnist.test.images, layer_num, batch_mask, False, reuse=True)
    test_shape = int_shape(mnist.test.images)
    test_mask = getMask(test_shape)
    x_to_z = sess.run(z, feed_dict={x_input:mnist.test.images, xs: test_shape, mask: test_mask, is_training:False, reuse:True})
    feed_dict = {z_input:x_to_z, labels: mnist.test.labels, is_training:False, reuse:True}
    print("test accuracy %g"%accuracy_Z.eval(feed_dict))
    
    ####################### Sample from Z space #######################
    logits = tf.placeholder(tf.float32, shape=[1, 10])
    # parameters
    num_steps=1000
    eps1 = 1e-3
    eps2 = 1e-8
    
    target_digit = 0
    target_label = np.reshape(np.array([int(i == target_digit) for i in range(10)]), (1,10))
    
    gradient = tf.gradients(cross_entropy_Z, z_input)
    z_0 = np.random.normal(0.0, 1.0, (sample_size,784)) # initialization of z
    
    # Initialization
    
    output_classifier = sess.run(output_Z_classifier, feed_dict={z_input:z_0, is_training:False, reuse:True})
    feed_dict = {z_input: z_0, labels: target_label, logits: output_classifier, is_training:False, reuse:True}
    class_likelihood = sess.run(gradient, feed_dict)[0]
    z_t = z_0 - eps1*class_likelihood + np.random.normal(0.0, eps2*1.0, (1, 784))
    output_classifier = sess.run(tf.nn.softmax(output_Z_classifier)[target_digit], feed_dict)[0]
    
    digits=[]
    for i in range(num_steps):
        output_classifier = sess.run(output_Z_classifier, feed_dict={z_input:z_t, is_training:False, reuse:True})
        feed_dict = {z_input: z_t, labels: target_label, logits: output_classifier, is_training:False, reuse:True}
        class_likelihood = sess.run(gradient, feed_dict)[0]
        z_t = z_0 - eps1*class_likelihood + np.random.normal(0.0, eps2*1.0, (1, 784))
        output_classifier = sess.run(tf.nn.softmax(output_Z_classifier)[target_digit], feed_dict)[0]
        
        
        
#        feed_dict = {z_input:z_t, labels:target_label, is_training:False, reuse:True}
#        class_likelihood = sess.run(gradient, feed_dict)[0]
#        z_t = z_t - eps1*class_likelihood + np.random.normal(0.0, eps2*1.0, (sample_size, 784))
#        #sample_accuracy = sess.run(accuracy_Z, feed_dict)
#        #cross_entropy = sess.run(cross_entropy_Z, feed_dict)
#        output_classifier = sess.run(tf.nn.softmax(output_Z_classifier)[target_digit], feed_dict)[0]
        if i%100 == 0:
            print(output_classifier)
        #print("step %d, sample accuracy %g, cross entropy %g"%(t, sample_accuracy, cross_entropy))
    
    # plot the z -> x
    z_to_x = tf.placeholder(tf.float32, shape=(sample_size,784))
    x_gen = inv_generator(z_to_x, layer_num)
    digit = sess.run(x_gen, {z_to_x:z_t})
    drawMNISTs(digit)   

    
    #sess.close()
    #del sess 
    
    
    
    
    
    
