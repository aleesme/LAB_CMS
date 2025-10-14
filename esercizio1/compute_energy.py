import matplotlib.pyplot as plt
import numpy as np
import sys
import math

def phi (rij) :
    '''
    this function receives the distance between two atoms and computes the Lennard-Jones potential
    '''
    epsilon = 0.345
    sigma = 2.644
    sum = (sigma/rij)**(12) - (sigma/rij)**(6)
    phi = 4*epsilon*sum
    return phi


def energy (filename):
    '''
    this function receives a file with the coordinates (x, y, z) of all atoms,
    performs a loop over the atoms to compute the distances and finally compute the potential energy
    '''

    file_path = "data/" + filename

    coord = np.loadtxt(file_path)
    N = coord.shape[0]
    E_pot = 0

    for i in range (N) :
        for j in range (i+1, N) :
            dx = coord[i, 0] - coord[j, 0]
            dy = coord[i, 1] - coord[j, 1]
            dz = coord[i, 2] - coord[j, 2]
            rij = np.sqrt(dx*dx+dy*dy+dz*dz)
            E_pot += phi(rij)

    return E_pot