import os, sys, re
import numpy as np
import time
from datetime import datetime
from matplotlib import pylab as plt
class TimedBlock:
    def __init__(self, name):
        self.name = name
        pass

    def __enter__(self):
        print "{}: Started {}.".format(datetime.now(), self.name)
        self.start_time = time.time()

    def __exit__(self, *args):
        print "{}: Finished {} in {:.2f} seconds.".format(datetime.now(), self.name, time.time() - self.start_time)

# A class to wrap scipy random variables to keep them centered
class CenteredRandomVariable:
    def __init__(self, dist):
        self.dist = dist

        self.name = "centered " + self.dist.name
        # find the index of the location parameter
        p = dist.fit(np.random.rand(1000,), floc=0)
        ind_loc = []
        for i,pval in enumerate(p):
            if pval == 0:
                ind_loc.append(i)
        if len(ind_loc) == 0:
            raise ValueError("Could not determine location of location parameter.")
        elif len(ind_loc) > 1:
            raise ValueError("Index of location parameter is ambiguous.")
        else:
            self.ind_loc = ind_loc[0]

        self.general2centered = lambda p: tuple(list(p[:self.ind_loc]) + list(p[(self.ind_loc+1):]))

        # How to get the params from centered into 
        self.centered2general = lambda p: tuple(list(p[:self.ind_loc]) + [0] + list(p[self.ind_loc:])) 

    def __str__(self):
        return self.name
        
    def fit(self, data):
        p = self.general2centered(self.dist.fit(data, floc=0))
        return p
    
    def cdf(self, vals, *params):
        return self.dist.cdf(vals, *(self.centered2general(params)))

    def pdf(self, vals, *params):
        return self.dist.pdf(vals, *(self.centered2general(params)))
    
class MixtureModel:
    def __init__(self, dists):
        self.dists = dists        
        self.names = [d.name for d in self.dists]
        self.bounds = []

    def _generate_random_parameters(self):
        p = []
        for b in self.bounds:
            if b[0] == -np.inf and b[1] == np.inf:
                p.append(np.random.randn())
            elif b[1] != np.inf:
                p.append(np.random.rand()*(b[1]-b[0]) + b[0])
            else:
                p.append(np.random.rand())
        p = np.array(p)
        p[-len(self.dists):] /= sum(p[-len(self.dists):])
        return p

    def _determine_num_dist_params(self, data):
        self.n_dist_params = [len(d.fit(data)) for d in self.dists]
        self.n_params = sum(self.n_dist_params) + len(self.dists)
        self.slices = reduce(lambda a, b: a + [slice(a[-1].stop, a[-1].stop+b)], self.n_dist_params[1:], [slice(0,self.n_dist_params[0])])
        return self.n_dist_params    
    
    def _determine_bounds(self, data):
        xvals = np.array(np.linspace(min(data), max(data),11))

        self.bounds = []        
        for i_dist, dist in enumerate(self.dists):
            # Setup a default set of parameters to perturb
            p0 = np.ones((self.n_dist_params[i_dist],)) 
            for i_param in range(self.n_dist_params[i_dist]):
                for val in [-1,0]:
                    p1 = 1.*p0
                    p1[i_param] = val
                    failed = False
                    if any(np.isnan(dist.pdf(data, *p1))):
                        failed = True
                        break
                self.bounds.append((0 if val == -1 else (1e-6 if failed else -np.inf), np.inf))

        # The parameters for the mixture weights
        for i in range(len(self.dists)): 
            self.bounds.append((0,1))

        return self.bounds

    def _determine_objective_function(self, data):
        nd = len(self.dists)
        #self.xvals = np.array([np.percentile(data, p) for p in range(0,101)])
        self.xvals = np.linspace(-max(data), max(data),101);
        self.yvals = np.array([np.mean(data<x) for x in self.xvals])
        self.cdf_fun = lambda p, x: np.dot(p[-nd:],np.stack([d.cdf(x, *p[self.slices[i]]) for i,d in enumerate(self.dists)]))        
        self.pdf_fun = lambda p, x: np.dot(p[-nd:],np.stack([d.pdf(x, *p[self.slices[i]]) for i,d in enumerate(self.dists)]))        
        self.obj_fun = lambda p: np.sum((self.cdf_fun(p,self.xvals) - self.yvals)**2)

    def _make_feasable(self, p):
        nd = len(self.dists)
        p1 = 1.*p;
        p1[-nd:] /= sum(p1[-nd:])
        return p1

    def _clamp_weights(self, p, mixture_weights):
        nd = len(self.dists)
        p1 = 1.*p
        p1[-nd:] = mixture_weights
        return p1

    def plot_fit_cdf(self, data):
        plt.fill_between(self.xvals, self.yvals,"gray",facecolor="gray",edgecolor="gray", label="data")
        plt.plot(self.xvals, self.cdf_fun(self.best, self.xvals), "k",label="fit")
        plt.legend()
        
    def plot_fit_pdf(self, data):
        h,b = np.histogram(data, int(np.sqrt(len(data))), density=True)
        b = (b[:-1] + b[1:])*0.5
        plt.fill_between(b, h, edgecolor="gray", facecolor="gray",label="data")
        for i,d in enumerate(self.dists):
            plt.plot(self.xvals, self.best[-len(self.dists)+i]*d.pdf(self.xvals, *self.best[self.slices[i]]), label = "{}: {}".format(d.name, self.best[self.slices[i]]))
        plt.plot(self.xvals, self.pdf_fun(self.best, self.xvals), "k", label="fit")
        plt.legend(facecolor=None, frameon=False)
    
    def fit(self, data, mixture_weights = [], **kwargs):
        self._determine_num_dist_params(data)
        self._determine_bounds(data)
        self._determine_objective_function(data)
        if mixture_weights and type(mixture_weights) is float:
            mixture_weights = [mixture_weights]*len(self.dists)
        if mixture_weights:
            if len(mixture_weights) != len(self.dists):
                raise ValueError("Needed exactly {} mixture weights, got {}.".format(len(self.dists), len(mixture_weights)))
            print "Clamping mixture weights to {}".format(mixture_weights)
        results = de(self.obj_fun, self.bounds, constraints = [lambda p: self._make_feasable(p)] if not mixture_weights else [lambda p: self._clamp_weights(p, mixture_weights)], **kwargs)
        self.best = results["best"]
        self.history = results["history"]
        print "FIT RESULTS"
        for i, d in enumerate(self.dists):
            print "{: 6.2f} x {} ({})".format(self.best[-len(self.dists)+i], self.names[i], self.best[self.slices[i]])
        return results
        
    
def de(obj_fun, bounds, constraints = [], n_iters = 1000, pop_size = 20, mut = 0.8, crossp = 0.7):
    """ Optimize OBJ_FUN using differential evolution.
    BOUNDS: A list of (min,max) tuples.
    CONSTRAINTS: A list of functions that project parameters into the feasable set. These should be
    constraints that aren't simple bounds.

    RETURNS:

    A dictionary with fields:
    'best': The best parameter values found.
    'history': The history of best values found after each iteration.

    Based with very slight modifications on the code at:
    https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/"""
    apply_constraints = lambda p: reduce(lambda pp, cons: cons(pp), constraints, p)
    
    n_dims = len(bounds)
    lbnds  = [b[0] for b in bounds]
    ubnds  = [b[1] for b in bounds]

    pop   = np.stack([generate_random_parameters(bounds, constraints) for i in range(pop_size)])

    fitness  = [obj_fun(p) for p in pop]
    best_idx = np.argmin(fitness)
    best     = pop[best_idx]
    history  = [min(fitness)]
    for i in range(n_iters):
        for j in range(pop_size):
            ids = range(j) + range(j+1,pop_size)
            a,b,c = pop[np.random.choice(ids, 3, replace=False)]
            mutant = apply_constraints(np.clip(a + mut*(b-c), lbnds, ubnds))
            cross_points = np.random.rand(n_dims) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, n_dims)] = True
            trial = apply_constraints(np.where(cross_points, mutant, pop[j]))
            f = obj_fun(trial)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = pop[best_idx]
        history.append(fitness[best_idx])
        if not (i % 100) or i == (n_iters-1):
            print "iter{:>6d}: {}".format(i, history[-1])
    return {"best":best, "history":history}

#def fit_mm_with_de(data, dists, prctiles = np.arange(0,101)):
