import numpy as np

def lp_metric(A, B, p):
    return np.sum(np.abs(A-B)**p, axis=-1) ** (1/p)

def karlsruhe(A, B):
    rA = np.linalg.norm(A, axis=-1)
    rB = np.linalg.norm(B, axis=-1)
    phiA = np.arctan2(A[:,0], A[:,1])
    phiB = np.arctan2(B[:,0], B[:,1])
    delta = np.minimum(np.abs(phiA-phiB),
                       2*np.pi-np.abs(phiA-phiB))

    case1 = np.minimum(rA, rB) * delta + np.abs(rA-rB)
    case2 = rA + rB

    return case1 * (delta <= 2) + case2 * (delta > 2)

def GCD(A, B, R):
    # convert to radians
    A *= (np.pi/180)
    B *= (np.pi/180)
    longA, latA = A.T
    longB, latB = B.T
    delta_long = longB-longA
    sA = np.sin(latA)
    sB = np.sin(latB)
    cA = np.cos(latA)
    cB = np.cos(latB)
    central_angle = np.arctan(np.sqrt((cB*np.sin(delta_long))**2
                                      + (cA*sB-sA*cB*np.cos(delta_long))**2)
                              / (sA*sB+cA*cB*np.cos(delta_long)))

    return R * central_angle
