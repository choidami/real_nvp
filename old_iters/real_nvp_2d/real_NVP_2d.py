from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.util import flatten
from autograd.optimizers import adam
from autograd.scipy.misc import logsumexp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def neural_net_predict(params, inputs):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return inputs

def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

def log_posterior(params, inputs, targets, L2_reg):
    log_prior = -L2_reg * l2_norm(params)
    log_lik = np.sum(neural_net_predict(params, inputs) * targets)
    return log_prior + log_lik

def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate):
    rads = np.linspace(0, 2*np.pi, num_classes, endpoint=False)

    features = npr.randn(num_classes*num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:,0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:,0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    return 10*npr.permutation(np.einsum('ti,tij->tj', features, rotations))

def logisticDistribution(h):
    return -np.log(1+np.exp(h)) - np.log(1+np.exp(-h))

def gaussianDistribution(h):
    return -0.5*(h**2 + np.log(2*np.pi))

def feed_forward(params, iter):
    idx = idxs[iter]
    inputs = data[idx].T
    x1 = inputs[0].reshape(inputs.shape[1], 1)
    x2 = inputs[1].reshape(inputs.shape[1], 1)
    
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

def inverse_flow(h1, h2, params):
    
    y1_layer2, y2_layer2 = h1, (h2 - neural_net_predict(params[5], h1))*np.exp(-neural_net_predict(params[4], h1))
    y1_layer1, y2_layer1 = (y2_layer2 - neural_net_predict(params[3], y1_layer2))*np.exp(-neural_net_predict(params[2], y1_layer2)), y1_layer2
    x1, x2 = y1_layer1, (y2_layer1 - neural_net_predict(params[1], y1_layer1))*np.exp(-neural_net_predict(params[0], y1_layer1))
   
    return x1, x2
    
def logLikelihood(params, iter):
    h1, h2, s = feed_forward(params,iter)
    reg = -L2_reg * l2_norm(params)
    return -np.sum(gaussianDistribution(h1) + gaussianDistribution(h2) + s)# - reg
    
# Generate fake mixture of gaussian
n_samples = 300

np.random.seed(0)

# generate spherical data centered on (20,20)
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20,20])

# generate stretched gaussian
C1 = np.array([[1., 0.0], [0.0, 5.0]])
stretched_gaussian1 = np.dot(np.random.randn(n_samples, 2), C1)

# generate zero centered stretched Gaussian data
C2 = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian2 = np.dot(np.random.randn(n_samples, 2), C2)

# concatenate the two datasets into the final training set 
#X_train = np.vstack([shifted_gaussian, stretched_gaussian1, stretched_gaussian2])
#X_train = stretched_gaussian1
num_clusters = 5
samples_per_cluster=2000
X_train = make_pinwheel_data(0.3, 0.05, num_clusters, samples_per_cluster, 0.25)

# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=num_clusters, covariance_type='full')
clf.fit(X_train)

# display predicted scores by the model as a contour plot
x = np.linspace(-20., 30.)
y = np.linspace(-20., 40.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()


# input data-set (fixed)
N = 10000
#data = clf.sample(N)[0]
data = X_train
#plt.scatter(data.T[0], data.T[1])
#plt.show()
#data = X_train

if __name__ == '__main__':
    ####### Simple real NVP #######
    # Training Parameters
    batch_size = 500 
    num_epoch =5000
    learning_rate = 0.05
    param_scale = 0.1
    layer_num = 3
    layer_sizes = [1, 30, 1]
    L2_reg = 1.0
    
    rs=npr.RandomState(0)
    
    # initialize the weights and biases for layer_num*2 neural nets- s and t's. 
    init_params = [init_random_params(param_scale, layer_sizes) for i in xrange(layer_num*2)] 
    #parameters = rs.rand(20) #[s1_w, s1_b, t1_w, t1_b, s2_w, s2_b, t2_w, t2_b, s3_w, s3_b, t3_w, t3_b]
    idxs = [np.random.randint(0, N, size=batch_size) for i in xrange(num_epoch)]
    
    grad_logLikelihood = grad(logLikelihood, argnum = 0)
    
    #h1, h2 = np.random.randn(batch_size,2).T # sample from a gaussian
    #x1, x2 = inverse_flow(h1, h2, parameters)
    #plt.scatter(h1, h2)
    #plt.show()
    
    log_likelihoods = []
        
    print("     Epoch     |    params   ")
    def print_logLikelihood(params, iter, gradient):
        log_likelihood = logLikelihood(params, iter)
        log_likelihoods.append(log_likelihood)
        if iter%100 == 0:
            h1, h2, s = feed_forward(params, iter)
            plt.scatter(h1, h2)
            plt.show()
            print("{:15}|{:20}".format(iter, log_likelihood))
            #h1, h2 = np.random.randn(batch_size,2).T
            #x1, x2 = inverse_flow(h1, h2, params)
            #plt.scatter(x1, x2)
            #plt.show()
        
    
    optimized_params = adam(grad_logLikelihood, init_params, step_size=learning_rate,
                                num_iters=num_epoch, callback=print_logLikelihood)
    x_axis = np.linspace(0,num_epoch, num_epoch)    
    plt.plot(x_axis, log_likelihoods)
    plt.show() 
    h1 = np.random.randn(batch_size,1)
    h2 = np.random.randn(batch_size,1)
    x1, x2 = inverse_flow(h1, h2, optimized_params)                        
    plt.scatter(x1, x2)    
    plt.show()    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    