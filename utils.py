import os, re
from functools import reduce

def progn(*args):
    for a in args:
        a()

chain = lambda af: lambda x: reduce(lambda a, f: f(a), af[1:],af[0](x))         

fapply = lambda farr, a: [f(a) for f in farr] # applies each function in farr to a

# Destructure a dictionary
def dd(*args): 
    return lambda d: [d[a] for a in args]

        
def expand_environment_variables(s, cleanup_multiple_slashes = True):
    for v in re.findall("(\$[A-Za-z_]+)", s):
        s = s.replace(v, os.environ[v[1:]]) # v[1:] to skip the $ at the beginning, so $HOME -> HOME
    if cleanup_multiple_slashes:
        s = re.sub("/+", "/", s)
    return s
