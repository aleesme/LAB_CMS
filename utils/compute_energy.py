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


def force (rkj, xj, xk) :
    '''
    this function compute the force between atoms k and j along the axis specified in the lasta arguments
    '''
    epsilon = 0.345
    sigma = 2.644
    F = 24 * epsilon * sigma**6 * rkj**(-8) * (xk-xj) * (2*(sigma/rkj)**6 - 1)
    return F


def energy (filename, rc=None):
    '''
    this function receives a file with the coordinates (x, y, z) of all atoms,
    performs a loop over the atoms to compute the distances and finally compute the potential energy
    if rc is not None, performs a cutoff
    '''

    file_path = "../data/" + filename

    coord = np.loadtxt(file_path)
    N = coord.shape[0]
    E_pot = 0

    if rc==None :
        for i in range (N) :
            for j in range (N) :
                if j != i :
                    dx = coord[i, 0] - coord[j, 0]
                    dy = coord[i, 1] - coord[j, 1]
                    dz = coord[i, 2] - coord[j, 2]
                    rij = np.sqrt(dx*dx+dy*dy+dz*dz)
                    E_pot += phi(rij)
    else :
        for i in range (N) :
            for j in range (N) :
                if j != i :
                    dx = coord[i, 0] - coord[j, 0]
                    dy = coord[i, 1] - coord[j, 1]
                    dz = coord[i, 2] - coord[j, 2]
                    rij = np.sqrt(dx*dx+dy*dy+dz*dz)
                    if rij < rc :
                        E_pot += phi(rij)

    return E_pot/2


def force_matrix (filename, rc=None) :
    '''
    this function returns a matrix containing the total force along x, y and z for each atom 
    '''
    file_path = "../data/" + filename

    coord = np.loadtxt(file_path)
    N = coord.shape[0]
    F_matrix = []

    if rc==None :
        for i in range (N) :
            Fx = Fy = Fz = 0.0
            for j in range (N) :
                if i != j:
                    dx = coord[i, 0] - coord[j, 0]
                    dy = coord[i, 1] - coord[j, 1]
                    dz = coord[i, 2] - coord[j, 2]
                    rij = np.sqrt(dx*dx+dy*dy+dz*dz)
                    Fx += force(rij, coord[j, 0] ,coord[i, 0])
                    Fy += force(rij, coord[j, 1] ,coord[i, 1])
                    Fz += force(rij, coord[j, 2] ,coord[i, 2])
            F_matrix.append([Fx, Fy, Fz])
    else :
        for i in range (N) :
            Fx = Fy = Fz = 0.0
            for j in range (N) :
                if i != j :
                    dx = coord[i, 0] - coord[j, 0]
                    dy = coord[i, 1] - coord[j, 1]
                    dz = coord[i, 2] - coord[j, 2]
                    rij = np.sqrt(dx*dx+dy*dy+dz*dz)
                    if rij < rc :
                        Fx += force(rij, coord[j, 0] ,coord[i, 0])
                        Fy += force(rij, coord[j, 1] ,coord[i, 1])
                        Fz += force(rij, coord[j, 2] ,coord[i, 2])
            F_matrix.append([Fx, Fy, Fz])

    return F_matrix


def nbrs (filename, rc) :
    '''
    this function computes the total number of neighbours for each atom in the file
    '''

    file_path = "../data/" + filename

    coord = np.loadtxt(file_path)
    N = coord.shape[0]
    n_i = 0
    n = []
    

    for i in range (N) :
        n_i = 0.0
        for j in range (N) :
            if j != i :
                dx = coord[i, 0] - coord[j, 0]
                dy = coord[i, 1] - coord[j, 1]
                dz = coord[i, 2] - coord[j, 2]
                rij = np.sqrt(dx*dx+dy*dy+dz*dz)
                if rij < rc :
                    n_i += 1
        n.append(n_i)

    return n


def which_nbrs (filename, rc) :
    '''
    this function return a list, each entrance i of the list contains an array with the atom numbers
    of the neighbours of the atom i
    '''

    file_path = "../data/" + filename

    coord = np.loadtxt(file_path)
    N = coord.shape[0]
    indices_matrix = []

    for i in range (N) :
        indices_i = []
        for j in range (N) :
            if j != i :
                dx = coord[i, 0] - coord[j, 0]
                dy = coord[i, 1] - coord[j, 1]
                dz = coord[i, 2] - coord[j, 2]
                rij = np.sqrt(dx*dx+dy*dy+dz*dz)
                if rij < rc :
                    indices_i.append(j)
        indices_matrix.append(indices_i)

    return indices_matrix


def force_matrix_nbrs (filename, rc) :
    '''
    this function returns a matrix containing the total force along x, y and z for each atom
    it computes the forces only between the neighbours, using the which_nbrs function 
    '''
    file_path = "../data/" + filename

    coord = np.loadtxt(file_path)
    N = coord.shape[0]
    F_matrix = []
    indices_matrix = which_nbrs(filename, rc)

    for i in range (N) :
        Fx = Fy = Fz = 0.0
        for j in indices_matrix[i]:
            dx = coord[i, 0] - coord[j, 0]
            dy = coord[i, 1] - coord[j, 1]
            dz = coord[i, 2] - coord[j, 2]
            rij = np.sqrt(dx*dx+dy*dy+dz*dz)
            Fx += force(rij, coord[j, 0] ,coord[i, 0])
            Fy += force(rij, coord[j, 1] ,coord[i, 1])
            Fz += force(rij, coord[j, 2] ,coord[i, 2])
        F_matrix.append([Fx, Fy, Fz])

    return F_matrix


def energy_nbrs (filename, rc) :
    '''
    this function receives a file with the coordinates (x, y, z) of all atoms,
    performs a loop over the atoms to compute the distances and finally compute the potential energy
    using only the neighbours
    '''

    file_path = "../data/" + filename

    coord = np.loadtxt(file_path)
    N = coord.shape[0]
    indices_matrix = which_nbrs(filename, rc)
    E_pot = 0

    for i in range (N) :
        for j in indices_matrix[i] :
            dx = coord[i, 0] - coord[j, 0]
            dy = coord[i, 1] - coord[j, 1]
            dz = coord[i, 2] - coord[j, 2]
            rij = np.sqrt(dx*dx+dy*dy+dz*dz)
            E_pot += phi(rij)

    return E_pot/2
