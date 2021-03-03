import numpy as np

def lp_metric(A, B, p):
    return np.sum(np.abs(A-B)**p, axis=-1) ** (1/p)

# def karlsruhe(A, B):


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
