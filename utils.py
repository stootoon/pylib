from functools import reduce

def progn(*args):
    for a in args:
        a()

chain = lambda af: lambda x: reduce(lambda a, f: f(a), af[1:],af[0](x))         

# Destructure a dictionary
def dd(*args): 
    return lambda d: [d[a] for a in args]

        
