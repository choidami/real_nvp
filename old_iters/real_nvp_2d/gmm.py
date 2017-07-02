import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
from autograd import grad
from autograd.util import flatten
from autograd.optimizers import adam
from autograd.scipy.misc import logsumexp
import random

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
    x1 = inputs[0]
    x2 = inputs[1]
    
    y1_layer1, y2_layer1 = x1, x2 + np.maximum(0, x1*params[0] + params[1])    
    y1_layer2, y2_layer2 = y2_layer1, y1_layer1 + np.maximum(0, y2_layer1*params[2] + params[3])  
    y1_layer3, y2_layer3 = y1_layer2, y2_layer2 + np.maximum(0, y1_layer2*params[4] + params[5])
    #y1_layer4, y2_layer4 = y2_layer3, y1_layer3 + np.maximum(0, y2_layer3*params[6] + params[7])
    
    #y1_layer1, y2_layer1 = x1, x2 + np.tanh(x1*params[0] + params[1])    
    #y1_layer2, y2_layer2 = y2_layer1, y1_layer1 + np.tanh(y2_layer1*params[2] + params[3])  
    #y1_layer3, y2_layer3 = y1_layer2, y2_layer2 + np.tanh(y1_layer2*params[4] + params[5])
    
    h1, h2 = np.exp(params[6])*y1_layer3, np.exp(params[7])*y2_layer3
                   
    return h1, h2

def inverse_flow(h1, h2, params):
    #idx = idxs[iter]
    #inputs = data[idx].T
    #x1= inputs[0]
    #x2 = inputs[1]
    
    y1_layer3, y2_layer3 = np.exp(-params[6])*h1, np.exp(-params[7])*h2
    y1_layer2, y2_layer2 = y1_layer3, y2_layer3 - np.maximum(0, y1_layer3*params[4] + params[5])
    y1_layer1, y2_layer1 = y2_layer2 - np.maximum(0, y1_layer2*params[2] + params[3]), y1_layer2
                                                 
    x1, x2 = y1_layer1, y2_layer1 - np.maximum(0, y1_layer1*params[0] + params[1])
    
    #y1_layer3, y2_layer3 = np.exp(-params[6])*h1, np.exp(-params[7])*h2
    #y1_layer2, y2_layer2 = y1_layer3, y2_layer3 - np.tanh(y1_layer3*params[4] + params[5])
    #y1_layer1, y2_layer1 = y2_layer2 - np.tanh(y1_layer2*params[2] + params[3]), y1_layer2
                                                 
    #x1, x2 = y1_layer1, y2_layer1 - np.tanh(y1_layer1*params[0] + params[1])
    
    
    return x1, x2
    
def logLikelihood(params, iter):
    h1, h2 = feed_forward(params,iter)
    #h = np.vstack((h1, h2))
        
    #return np.sum(logisticDistribution(h1) + logisticDistribution(h2) + params[6] + params[6])
    return -np.sum(gaussianDistribution(h1) + gaussianDistribution(h2) + params[6] + params[7])
    
# Generate fake mixture of gaussian
n_samples = 300

np.random.seed(0)

# generate spherical data centered on (20,20)
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20,20])

# generate stretched gaussian
C1 = np.array([[1., 0.0], [0.0, 5.0]])
stretched_gaussian1 = np.dot(np.random.randn(n_samples, 2) + np.array([0,2]), C1)

# generate zero centered stretched Gaussian data
C2 = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian2 = np.dot(np.random.randn(n_samples, 2), C2)

# concatenate the two datasets into the final training set 
#X_train = np.vstack([shifted_gaussian, stretched_gaussian1, stretched_gaussian2])t
X_train = stretched_gaussian2
num_clusters = 1
#samples_per_cluster=2000
#X_train = make_pinwheel_data(0.3, 0.05, num_clusters, samples_per_cluster, 0.25)

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
data = clf.sample(N)[0]
#plt.scatter(data.T[0], data.T[1])
#plt.show()
#data = X_train
####### Simple NICE #######

# Training Parameters
batch_size = 500 
num_epoch =100
learning_rate = 0.05

rs = np.random.RandomState(10)
parameters = rs.randn(8) #[w1, b1, w2, b2, w3, b3, s1, s2]
idxs = [np.random.randint(0, N, size=batch_size) for i in xrange(num_epoch)]

grad_logLikelihood = grad(logLikelihood, argnum = 0)

h1, h2 = np.random.randn(batch_size,2).T
#h1, h2 = feed_forward(parameters, 0)
x1, x2 = inverse_flow(h1, h2, parameters)
plt.scatter(h1, h2)
plt.show()
    
log_likelihoods = []

print("     Epoch     |    params   ")
def print_logLikelihood(params, iter, gradient):
    log_likelihood = logLikelihood(params,iter)
    h1, h2 = feed_forward(params, iter)
    #h = np.vstack((h1, h2))
    #plt.scatter(h1, h2)
    #plt.show()
    log_likelihoods.append(log_likelihood)
    print("{:15}|{:20}".format(iter, params))
    h1, h2 = np.random.randn(batch_size,2).T
    x1, x2 = inverse_flow(h1, h2, params)
    plt.scatter(x1, x2)
    plt.show()
    
    
optimized_params = adam(grad_logLikelihood, parameters, step_size=learning_rate,
                            num_iters=num_epoch, callback=print_logLikelihood)

x_axis = np.linspace(0,num_epoch, num_epoch)    
plt.plot(x_axis, log_likelihoods)

#for i in xrange(num_epoch):
#    # sample N points from data
#    mini_batch_idx = np.random.randint(0, 1000, size=batch_size)
#    #mini_batch = np.array(random.sample(data,batch_size)).T
#    
#    d_param = np.zeros(8)  
#    
#    log_likelihood = logLikelihood()
#    
#    for j in xrange(batch_size):
#
#        log_likelihood = logLikelihood(x1[j], x2[j], w1, w2, w3, s1, s2, b1, b2, b3)
#                                              
#        grad_w1 = grad(logLikelihood, argnum=2)
#        grad_w2 = grad(logLikelihood, argnum=3)
#        grad_w3 = grad(logLikelihood, argnum=4)
#        grad_s1 = grad(logLikelihood, argnum=5)
#        grad_s2 = grad(logLikelihood, argnum=6)
#        grad_b1 = grad(logLikelihood, argnum=7)
#        grad_b2 = grad(logLikelihood, argnum=8)
#        grad_b3 = grad(logLikelihood, argnum=9)
#        
#        d_w1 += grad_w1(0,0,w1,0,0,0,0,0,0,0)
#        d_w2 += grad_w1(0,0,0,w2,0,0,0,0,0,0)
#        d_w3 += grad_w1(0,0,0,0,w3,0,0,0,0,0)
#        d_s1 += grad_w1(0,0,0,0,0,s1,0,0,0,0)
#        d_s2 += grad_w1(0,0,0,0,0,0,s2,0,0,0)
#        d_b1 += grad_w1(0,0,0,0,0,0,0,b1,0,0)
#        d_b2 += grad_w1(0,0,0,0,0,0,0,0,b2,0)
#        d_b3 += grad_w1(0,0,0,0,0,0,0,0,0,b3)
#        
#    d_w1 /= N
#    d_w2 /= N
#    d_w3 /= N
#    d_s1 /= N
#    d_s2 /= N
#    d_b1 /= N
#    d_b2 /= N
#    d_b3 /= N
#
#    w1 = w1 + learning_rate*d_w1
#    w2 = w2 + learning_rate*d_w2
#    w3 = w3 + learning_rate*d_w3
#    s1 = s1 + learning_rate*d_s1
#    s2 = s2 + learning_rate*d_s2
#    b1 = b1 + learning_rate*d_b1
#    b2 = b2 + learning_rate*d_b2
#    b3 = b3 + learning_rate*d_b3
    
    
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    