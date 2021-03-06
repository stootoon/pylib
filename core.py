import os, sys, re, types
import numpy as np
import time
from datetime import datetime
from matplotlib import pylab as plt
from scipy import fft
from collections import namedtuple
from pylib.utils import progn
from functools import reduce

import logging
class Tokens:
    def __init__(self, tokens):
        self.tokens = tokens

    def any(self, needles):
        return len(set(needles) & set(self.tokens))>0        

    def all(self, needles):
        return len(set(needles) & set(self.tokens)) == len(needles)

    def __contains__(self, item):
        return item in self.tokens
        
def matplotlib_whisperer(text):

    tokens = text.split(" ")

    T = Tokens(tokens)

    action = None
    location = None
    obj = None

    if T.any(["set", "on", "show"]):
        action = "set"
    
    if T.any(["no", "remove", "delete", "off", "hide"]):
        action = "remove"

    if T.any(["bottom", "below", "lower"]):
        location = "bottom"

    if "left" in T:
        location = "left"
        
    if "right" in T:
        location = "right"
        
    if T.any(["top", "above", "upper"]):
        location = "top"

    if T.all(["legend", "background"]):
        obj = "legend_background"
        
    if T.all(["x", "ticks"]) or T.any(["x-ticks", "xticks"]):
        if "minor" in T:
            obj = "xminortticks"
        else:
            obj = "xticks"

    if "ticks" in T and "x" not in T and "y" not in T:
        if "minor" in T:
            obj = "minorticks"
        else:
            obj = "ticks"

    if "grid" in T and "x" not in T and "y" not in T:
        if "minor" in T:
            obj = "minorgrid"
        else:
            obj = "grid"
            
    if T.all(["y", "ticks"]) or T.any(["y-ticks", "yticks"]):
        if "minor" in T:
            obj = "yminorticks"
        else:
            obj = "yticks"

    if T.all(["x", "grid"]):
        if "minor" in T:
            obj = "xminorgrid"
        else:
            obj = "xgrid"

    if T.all(["y", "grid"]):
        if "minor" in T:
            obj = "yminorgrid"
        else:
            obj = "ygrid"
            
    if T.all(["x", "tick", "labels"]) or T.any(["xticklabels"]):
        obj = "xticklabels"
        
    if T.all(["y", "tick", "labels"]) or T.any(["yticklabels"]):
        obj = "yticklabels"

    ax = plt.gca()
    logging.basicConfig()
    log = logging.getLogger("whisperer")
    log.debug("  action: '{}'".format(action))
    log.debug("  object: '{}'".format(obj))
    log.debug("location: '{}'".format(location))
    maps = [
        {"action":[None, "set"], "obj":"xticks", "location":"bottom",  "do": lambda: ax.xaxis.set_ticks_position("bottom"), "msg": "Set xticks position to bottom."},
        {"action":[None, "set"], "obj":"xticks", "location":"top",     "do": lambda: ax.xaxis.set_ticks_position("top"),    "msg": "Set xticks position to top."},
        {"action":[None, "set"], "obj":"yticks", "location":"left",    "do": lambda: ax.yaxis.set_ticks_position("left"),   "msg": "Set yticks position to top."},
        {"action":[None, "set"], "obj":"yticks", "location":"right",   "do": lambda: ax.yaxis.set_ticks_position("right"),  "msg": "Set yticks position to right."},
        {"action":"set",    "obj":"minorgrid", "location":None,    "do": lambda: plt.grid(True, which="minor", axis="both"), "msg": "Showing minor grid."},
        {"action":"set",    "obj":"yminorgrid", "location":None,   "do": lambda: plt.grid(True, which="minor", axis="y"), "msg": "Showing minor y grid."},
        {"action":"set",    "obj":"xminorgrid", "location":None,   "do": lambda: plt.grid(True, which="minor", axis="x"), "msg": "Showing minor x grid."},        
        {"action":"remove",  "obj":"yticks", "location":None,   "do": lambda: ax.yaxis.set_ticks([]),       "msg": "Removed yticks."},
        {"action":"remove",  "obj":"yticks", "location":None,   "do": lambda: ax.yaxis.set_ticklabels([]),  "msg": "Removed ytick labels."},        
        {"action":"remove",  "obj":"xticks", "location":None,   "do": lambda: ax.xaxis.set_ticks([]),       "msg": "Removed xticks."},
        {"action":"remove",  "obj":"xticks", "location":None,   "do": lambda: ax.xaxis.set_ticklabels([]),  "msg": "Removed xtick labels."},
        {"action":"remove",  "obj":"legend_background", "location":None, "do":
         lambda: progn(
             lambda: plt.legend().get_frame().set_facecolor('none'),
             lambda: plt.legend().get_frame().set_linewidth(0)),
         "msg": "Removed legend background."},
    ]

    def check_match(m):
        mm = {f: v if type(m[f]) is list else [m[f]] for f,v in m.items()}
        if action not in mm["action"]:
            return False
        if obj not in mm["obj"]:
            return False
        if location not in mm["location"]:
            return False
        return True
        
    # Find matches
    for m in maps:
        if check_match(m):
            m["do"]()
            log.info(m["msg"])
            return
    log.warning("No actions found for {}: ".format(text))
            
def rms_error(x,y, verbose = True):
    err = np.sqrt(np.mean((x-y)**2))
    if verbose:
        print("RMS error: {}".format(err))

    return err

def spectrum(x, fs = 1., color = None, plot_fun = None, mean_subtract = False, mark_peak = False):
    if mean_subtract:
        x -= np.mean(x)
    f = fft(x)
    freqs = np.arange(len(x))/float(len(x))*fs
    if mark_peak:
        print("Peak AC frequency: {:.1f} Hz".format(freqs[(freqs>0)&(freqs<0.5*fs) ][np.argmax(abs(f[(freqs>0)&(freqs<0.5*fs)]))]))
    if plot_fun:
        plot_fun(freqs, abs(f), color=color) if color else plot_fun(freqs, abs(f))
        plt.xlim(0,0.5*fs)
        plt.xlabel("frequency / Hz")
        plt.ylabel("|X(f)|")
    return f, freqs

def find_runs(f):
    runs = []
    in_run = False
    for i,fi in enumerate(f):
        if in_run:
            if fi:
                this_run.append(i)
            else:
                runs.append(this_run)
                in_run = False
        else:
            if fi:
                this_run = [i]
                in_run = True
    else:
        if in_run:
            runs.append(this_run)

    return runs                       

class TimedBlock:
    def __init__(self, name):
        self.name = name
        pass

    def __enter__(self):
        print("{}: Started {}.".format(datetime.now(), self.name))
        self.start_time = time.time()

    def __exit__(self, *args):
        print("{}: Finished {} in {:.2f} seconds.".format(datetime.now(), self.name, time.time() - self.start_time))

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

    def plot_fit_cdf(self, data = None, color="gray"):
        if not data:
            data = self.data
        plt.fill_between(self.xvals, self.yvals,facecolor=color,edgecolor=color, label="data")
        plt.plot(self.xvals, self.cdf_fun(self.best, self.xvals), "k",label="fit")
        plt.legend()
        
    def plot_fit_pdf(self, data = None, color="gray"):
        if not data:
            data = self.data
        h,b = np.histogram(data, int(np.sqrt(len(data))), density=True)
        b = (b[:-1] + b[1:])*0.5
        plt.fill_between(b, h, edgecolor=color, facecolor=color,label="data")
        for i,d in enumerate(self.dists):
            param_str = ", ".join(["{:.3f}".format(f) for f in self.best[self.slices[i]]])
            plt.plot(self.xvals, self.best[-len(self.dists)+i]*d.pdf(self.xvals, *self.best[self.slices[i]]), label = "{}: ({})".format(d.name, param_str))
        plt.plot(self.xvals, self.pdf_fun(self.best, self.xvals), "k", label="fit")
        plt.legend(facecolor=None, frameon=False)
    
    def fit(self, data, mixture_weights = [], **kwargs):

        self.data = data

        if "constraints" in kwargs:
            raise ValueError("Named arguments to fit can't contain an entry for 'constraints'.")
        
        self._determine_num_dist_params(data)
        self._determine_bounds(data)
        self._determine_objective_function(data)

        # Set up the constraints based on the argument supplied as mixture_weight
        if mixture_weights:
            if type(mixture_weights) is types.FunctionType: # It was a function, use it directly
                constraint = mixture_weights
            elif type(mixture_weights) is float:
                if not np.allclose(mixture_weights*len(self.dists), 1):
                    mixture_weights = [mixture_weights] + [(1. - mixture_weights)/(len(self.dists)-1)]*(len(self.dists)-1)
                else:
                    mixture_weights = [mixture_weights]*len(self.dists) 
                constraint = lambda p: self._clamp_weights(p, mixture_weights)
            else:
                if len(mixture_weights) != len(self.dists):
                    raise ValueError("Number of mixture weights {} did not equal number of distributions {}.".format(len(mixture_weights), len(self.dists)))
                if not np.allclose(sum(mixture_weights), 1):
                    raise ValueError("Mixture weights must sum to 1, but actually sum to {}".format(sum(mixture_weights)))
                constraint = lambda p: self._clamp_weights(p, mixture_weights)
        else:
            constraint = lambda p: self._make_feasable(p)

        results = de(self.obj_fun, self.bounds, self._generate_random_parameters, constraints = [constraint], **kwargs)
        self.best = results["best"]
        self.history = results["history"]
        print("FIT RESULTS")
        for i, d in enumerate(self.dists):
            print("{: 6.2f} x {} ({})".format(self.best[-len(self.dists)+i], self.names[i], self.best[self.slices[i]]))
        return results
        
    
def de(obj_fun, bounds, generate_random_parameters, constraints = [], n_iters = 1000, pop_size = 20, mut = 0.8, crossp = 0.7):
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

    pop   = np.stack([generate_random_parameters() for i in range(pop_size)])

    fitness  = [obj_fun(p) for p in pop]
    best_idx = np.argmin(fitness)
    best     = pop[best_idx]
    history  = [min(fitness)]
    for i in range(n_iters):
        for j in range(pop_size):
            ids = list(range(j)) + list(range(j+1,pop_size))
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
            print("iter{:>6d}: {}".format(i, history[-1]))
    return {"best":best, "history":history}

#def fit_mm_with_de(data, dists, prctiles = np.arange(0,101)):
