"""Support function for the complex Bingham distribution
needed for Bayesian BSS,
to model the distribution of complex-valued microphone gain values.

Arne Leijon, 2014-07-12
2015-03-27 allow general array-like input
"""

import numpy as np
from scipy.optimize import brentq #root of non-linear equation

def logPartition_saddle( eigVV ):
    """Approximate log partition factor (normalization constant),
and the gradient of the log partition factor,
for one or more Bingham-distributed real-valued random vectors.

Input:
eigVV= array-like object containing vectors of real-valued eigenvalues,
    eigVV[...,j]= j-th eigenvalue
    for the [...]-th real-valued Bingham distribution.

Result: tuple (logC, DlogC),
logC= array with scalar log partition factors
    logC[...]= log partition factor log C(eigV[...,:])
    logC.shape == np.array(eigVV).shape[:-1]
DlogC= array with corresponding gradient vectors
    DlogC[...,k]= partial derivative: d log C(eigV[...,:]) / d eigVV[...,k]
    DlogC.shape == np.array(eigVV).shape

Method:
A "saddle-point" approximation was proposed by
Kume, Wood (2005). Biometrica 92(2), 465-476.
They claimed that this approximation is very good, and
can be seen as a way to implement l'Hopital's rule to handle
the multiplicity problems in the formulation of Kent(1994).

NOTE: This implementation uses only the first-order saddle-point approx.
It might be improved by higher order....

Arne Leijon, 2016-02-22
"""
    minLambda = 0.1 #constant for all saddle-point approximations
    eigVVshape = np.asarray(eigVV).shape
    #should work even if eigVV is only array-like
    p = eigVVshape[-1]#dimensionality of distributions
    eigV = np.reshape(eigVV,(-1, p))
##    print('eigVVshape:',eigVVshape)
##    print('eigVshape:',eigV.shape)

    Lamb = - eigV #Kume require eigenvalues of opposite sign to the input
    lAdjust = minLambda - np.min( Lamb, axis=-1,keepdims=True)
    # needed to make smallest Lamb positive.
    Lamb = Lamb + lAdjust
    #positive eigenvalues to be used in the saddle-point approx

    #------------------------------------------- internal sub-function:
    def SolveK1equal1(Lamb):
        """Solve equation K1(t) == 1; Kume&Wood eq 9
repeatedly for may vectors of eigenvalues.

Input:
Lamb= array of eigenvalue vectors, each with p elements,
using notation in Kume&Wood(2005)

Result:
t= column array of solutions , such that
K1( t[k,0], Lamb[k,:] ) == 1,

Arne Leijon, 2016-02-21
"""
    #------------------------------ equation to be solved:
        def K1minus1(t, Lambda_k):
            """Left-hand side of equation K1(t) - 1 == 0,
where K1(t) is first derivative in
Kume & Wood (2005) eq. (9)
for a single external eigenvalue vector

Input:
t= scalar variable
Lambda_k= 1D array of eigenvalues for a single Bingham distribution
Result = value of function K1(t)-1
to be used by equation solver.
"""
            return 0.5 * np.sum( 1/(Lambda_k - t) ) - 1.
    #----------------------------------------------------

        tMax = minLambda - 0.5
        tMin = minLambda - p #/ 2
        #bracketing interval for solution
        #Kume&Wood(2005) between (11) and (12)

        tHat = np.array( [brentq( K1minus1, tMin, tMax, args=(Lambda_k) )
                         for Lambda_k in Lamb ]
                        )
        return tHat[:,np.newaxis] #to allow broadcast

#-----------------------------------------------------------------
    tHat = SolveK1equal1(Lamb) # tHat satisfies K1(tHat) == 1; Kume eq 9
    #tHat[k,0]= t-value for Lamb[k,:], to allow broadcast

    K2 = 0.5 * np.sum( (Lamb - tHat)**-2, axis=-1, keepdims=True )
    #Kume eq 10, evaluated at tHat
    K3 = np.sum( (Lamb - tHat)**-3, axis=-1, keepdims=True )
    #Kume eq 10, evaluated at tHat
##K4 = 12 * sum( (Lamb - tHat)**-4);%Kume eq 10, evaluated at tHat
##
##rho3= K3/(K2.^1.5);
##rho4= K4/(K2.^2); %to be used in Kume eq 13
##
##T= rho4/8 - rho3.^2 * 5/24; %Kume eq 13
    T = 0.
    #****** using only first-order saddle approx for now,
    #because the derivative d T / d Lamb gets complicated for higher order

    logC1= ( 0.5*( np.log(2)+ (p-1) * np.log(np.pi) - np.log(K2)
                   - np.sum( np.log(Lamb - tHat), axis=-1, keepdims=True)
                   )
             - tHat
             )
    #Kume eq (15)

    #--- calc gradient d(logC) / dLamb, evaluated at given Lamb and tHat
    #**** using only first-order saddle-point approximation for now

    dK1dLamb = - 0.5 * (Lamb- tHat)**-2
    #= gradient d K1(Lamb) / d Lamb
    dK1dt = - np.sum(dK1dLamb, axis=-1, keepdims= True)
    #= d K1(Lamb) / d t
    dtdLamb = - dK1dLamb / dK1dt
    #= grad d t / d Lamb, for constant K1==1, evaluated at tHat
    dK2dLamb = - (Lamb-tHat)**-3 + K3 * dtdLamb
    #= gradient d K2(Lamb) / d Lamb, from eq 10, evaluated at tHat
    dlogK2dLamb = dK2dLamb / K2
    #= gradient d log K2(Lamb) / d Lamb
    dSumlogdLamb = 1. / (Lamb - tHat)
    #= gradient d sum(log(Lamb - tHat)) / d Lamb
    dSumlogdtHat = - np.sum(dSumlogdLamb, axis=-1, keepdims=True)
    #= d sum(log(Lamb - tHat))  / d tHat
    dSumlogdLamb = dSumlogdLamb + dSumlogdtHat * dtdLamb
    #= total gradient
    #DlogC = - 0.5* dlogK2dLamb - 0.5 * dSumlogdLamb - dtdLamb
    #= total d logC/ dLamb eq (15), but here
    DlogC = 0.5 * dlogK2dLamb + 0.5 * dSumlogdLamb + dtdLamb
    #we take opposite sign, because we used negated Lamb.
    #---------------------------------------------------------------------

    logC1 += lAdjust
    #compensate back for earlier lAdjust change

    #restore input shapes [...] and [...,:]
##    print('logC1.shape:', logC1.shape)
    logC1 = np.reshape(logC1, eigVVshape[:-1])
##    print('logC1.shape:', logC1.shape)
    DlogC = np.reshape(DlogC, eigVVshape)
    return ( logC1, DlogC )