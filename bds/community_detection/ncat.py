"""
Author: Shi Gu
This is the python edition of the network toolbox
Each author for the matlab version is in the doc string of each function
"""

import numpy as np
from scipy import sparse
from numpy import linalg
from scipy.spatial import ConvexHull
import sys
import random


def zrand(part1,part2):
    '''
    ZRAND     Calculates the z-Rand score and Variation of Information
    distance between a pair of partitions.

    [zRand,SR,SAR,VI] = ZRAND(part1,part2) calculates the z-score of the
    Rand similarity coefficient between partitions part1 and part2. The
    Rand similarity coefficient is an index of the similarity between the
    partitions, corresponding to the fraction of node pairs identified the
    same way by both partitions (either together in both or separate in
    both)

    NOTE: This code requires genlouvain.m to be on the MATLAB path

    Inputs:     part1,  | Partitions that are being
                part2,  | compared with one another

    Outputs:    zRand,  z-score of the Rand similarity coefficient
                SR,     Rand similarity coefficient
                SAR,    Adjusted Rand similarity coefficient
                VI,     Variation of information

    Amanda L. Traud, Eric D. Kelsic, Peter J. Mucha, and Mason A. Porter,
    "Comparing Community Structure to Characteristics in Online Collegiate
    Social Networks," SIAM Review 53, 526-543 (2011).
    '''
    if part1.shape[0] == 1:
        part1 = part1.T
    if part2.shape[0] == 1:
        part2 = part2.T
    if part1.shape != part2.shape:
        print('ERROR: partitions not of equal length')
        return
    '''
    Generate contingency table and calculate row/column marginals
    '''
    part1 = np.asarray(part1).flatten()
    part2 = np.asarray(part2).flatten()
    partSize1 = len(set(part1)) + 1
    partSize2 = len(set(part2)) + 1
    nij=sparse.csr_matrix((np.ones(part1.shape, dtype=int),(part1,part2)), shape=(partSize1, partSize2))
    ni=nij.sum(axis=1)
    nj=nij.sum(axis=0)
    nj=nj.T

    # Identify total number of elements, n, numbers of pairs, M, and numbers of
    # classified-same pairs in each partition, M1 and M2.

    n = part1.shape[0]
    M = np.double(n*(n-1)/2)
    M1 = np.double(((np.multiply(ni,ni)-ni)/2).sum())
    M2 = np.double(((np.multiply(nj,nj)-nj)/2).sum())

    # Pair counting types:

    # same in both
    a = ((nij.multiply(nij)-nij)/2).sum()
    # same in 1, diff in 2'
    b = M1-a
    # same in 2, diff in 1'
    c = M2-a
    # diff in both'
    d = M-(a+b+c)


    # Rand and Adjusted Rand indices:

    SR=(a+d)/(a+b+c+d)
    meana=M1*M2/M
    SAR=(a-meana)/((M1+M2)/2-meana)

    # PS: The adjusted coefficient is calculated by subtracting the expected
    # value and rescale the result by the difference between the maximum
    # allowed value and the mean value

    # CALCULATE VARIANCE OF AND Z-SCORE OF Rand
    # C2=sum(nj.^3);
    # C2=sum(nj.^3);
    # vara = (C1*C2*(n+1) - C1*(4*M2^2+(6*n+2)*M2+n^2+n) - C2*(4*M1^2+(6*n+2)*M1+n^2+n))...
    #     /(n*(n-1)*(n-2)*(n-3)) + M/16 - (4*M1-2*M)^2*(4*M2-2*M)^2/(256*M^2) +...
    #     (8*(n+1)*M1-n*(n^2-3*n-2))*(8*(n+1)*M2-n*(n^2-3*n-2))/(16*n*(n-1)*(n-2)) +...
    #    (16*M1^2-(8*n^2-40*n-32)*M1+n*(n^3-6*n^2+11*n+10))*...
    #     (16*M2^2-(8*n^2-40*n-32)*M2+n*(n^3-6*n^2+11*n+10))/(64*n*(n-1)*(n-2)*(n-3));

    C1=4*((np.power(ni,3)).sum())-8*(n+1)*M1+n*(n*n-3*n-2)
    C2=4*((np.power(nj,3)).sum())-8*(n+1)*M2+n*(n*n-3*n-2)

    # Calculate the variance of the Rand coefficient (a)

    vara = M/16 - np.power((4*M1-2*M),2)*np.power((4*M2-2*M),2)/(256*M*M) + C1*C2/(16*n*(n-1)*(n-2)) + \
    (np.power((4*M1-2*M),2)-4*C1-4*M)*(np.power((4*M2-2*M),2)-4*C2-4*M)/(64*n*(n-1)*(n-2)*(n-3))

    # Calculate the z-score of the Rand coefficient (a)

    zRand=(a-meana)/np.sqrt(vara)


    # CALCULATE THE VARIATION OF INFORMATION

    c1=set(part1);
    c2=set(part2);
    H1=0; H2=0; I=0;
    for i in c1:
        pi=np.double(ni[i])/n
        H1=H1-pi*np.log(pi)
        for j in c2:
            if nij[i,j]:
                pj=np.double(nj[j])/n;
                pij=np.double(nij[i,j])/n;
                I=I+pij*np.log(pij/pi/pj)
    for j in c2:
        pj=np.double(nj[j])/n
        H2=H2-pj*np.log(pj)
    VI = (H1+H2-2*I)
    return (zRand,SR,SAR,VI)


