import numpy as np
from scipy.stats import norm
import pymc3 as pm

# Real parameters
v1 = 25
v2 = -30
vec = np.array([v1, v2])

def pureModel(vec1,vec2):
    #result = np.dot(vec1,vec2)
    result = vec1 + vec2
    return result

def add_gauss_noise(data, mu=0, sd=1):
    return data + norm.rvs(size = data.size, loc=mu, scale=sd)

def rand_2D_vec(n=1):
    while n > 0:
        yield np.random.rand(2)*100
        n -= 1

nsamples = 100
datax = list(rand_2D_vec(nsamples))
datay = [add_gauss_noise(pureModel(x,vec)) for x in datax]

with pm.Model() as pmodel:
    # Define priors
    vec = pm.Normal('vec', mu=0,sd=1, shape=2)

    # Define likelihood
    #likelihood = Normal('y', mu= b + m * data.x.values, sd=sigma, observed=data.y.values)
    likelihood = pm.Normal('Something', mu = pureModel(datax,vec), sd=1, observed=datay)

    # Inference!
    start = pm.find_MAP()  # Find starting value by optimization
    step = pm.NUTS(scaling=start)  # Instantiate MCMC sampling algorithm
    trace = pm.sample(20, step, start=start, progressbar=True) # draw 2000 posterior samples using NUTS sampling
    
pm.traceplot(trace)

