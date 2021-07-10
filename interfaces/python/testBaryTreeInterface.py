'''
'''
import os
import sys
import resource
import numpy as np
import mpi4py.MPI as MPI
import time


sys.path.append(os.getcwd())
try:
    import BaryTreeInterface as BT
except ImportError:
    print('Unable to import BaryTreeInterface due to ImportError')
except OSError:
    print('Unable to import BaryTreeInterface due to OSError')


def runBary(num):
    # set treecode parameters
    N = num
    critical = 2000
      
    maxPerSourceLeaf = critical
    maxPerTargetLeaf = critical
    GPUpresent = True
    theta = 0.7
    treecodeDegree = 7
    gaussianAlpha = 0.7
    verbosity = 0
    
    approximation = BT.Approximation.LAGRANGE
    singularity   = BT.Singularity.SUBTRACTION
    computeType   = BT.ComputeType.CLUSTER_CLUSTER
    
    kernel = BT.Kernel.COULOMB
    numberOfKernelParameters = 1
    kernelParameters = np.array([0.5])


    # initialize some random data
    np.random.seed(1)
    RHO = np.random.rand(N)
    X = np.random.rand(N)
    Y = np.random.rand(N)
    Z = np.random.rand(N)
    W = np.ones(N)   # W stores quadrature weights for convolution integrals.  For particle simulations, set = ones.
    X2 = np.copy(X)
    Y2 = np.copy(Y)
    Z2 = np.copy(Z)
    RHO2 = np.copy(RHO)
    W2 = np.copy(W)

    # call the treecode
    start_time = time.time()
    output = BT.callTreedriver(  N, N,
                                 X, Y, Z, RHO,
                                 X2, Y2, Z2, RHO2, W2,
                                 kernel, numberOfKernelParameters, kernelParameters,
                                 singularity, approximation, computeType,
                                 GPUpresent, verbosity, 
                                 theta=theta, degree=treecodeDegree, sourceLeafSize=maxPerSourceLeaf, targetLeafSize=maxPerTargetLeaf, sizeCheck=1.0)
    print('-'*10 + ' bodies:{} '.format(num) + '-'*10)  # print divider between iterations
    print("--- time:%s ---" % (time.time() - start_time))
    
for i in range(25):
    numParticles = int(pow(10,(i+32)/8.0))
    runBary(numParticles)

