#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

class WaveFunction(object):

    def __init__(self, x, y, psi_0, V, dt, hbar=1, m=1, t0=0.0):
        self.x = np.array(x)
        self.y = np.array(y)
        self.psi = np.array(psi_0, dtype=np.complex128)
        self.V = np.array(V, dtype=np.complex128)
        self.dt = dt
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.hbar = hbar
        self.m = m
        self.t = t0

        alpha = dt/(4*self.dx**2)
        self.alpha = alpha
        self.size_x = len(x)
        self.size_y = len(y)
        dimension = self.size_x*self.size_y

        #Building the first matrix to solve the system (A from Ax_{n+1}=Mx_{n})
        N = (self.size_x-1)*(self.size_y-1)
        size = 5*N + 2*self.size_x + 2*(self.size_y-2)
        I = np.zeros(size)
        J = np.zeros(size)
        K = np.zeros(size, dtype=np.complex128)

        k = 0
        for i in range(0,self.size_y):
            for j in range(0,self.size_x):
                #Condition aux frontières nulles aux extrémités (en y)
                if i==0 or i==(self.size_y-1):
                    I[k] = i + j*self.size_y
                    J[k] = i + j*self.size_y
                    K[k] = 1
                    k += 1

                #Conditions aux frontières nulles aux extrémités (en x)
                elif j==0 or j==(self.size_x-1):
                    I[k] = i + j*self.size_y
                    J[k] = i + j*self.size_y
                    K[k] = 1
                    k += 1

                #Points à l'intérieur du domaine
                else:
                    #Point central (i,j)
                    I[k] = i + j*self.size_y
                    J[k] = i + j*self.size_y
                    K[k] = 1.0j - 4*alpha - V[i+j*self.size_y]*dt/2
                    k += 1

                    #Point (i-1,j)
                    I[k] = i + j*self.size_y
                    J[k] = (i-1) + j*self.size_y
                    K[k] = alpha
                    k += 1

                    #Point (i+1,j)
                    I[k] = i + j*self.size_y
                    J[k] = (i+1) + j*self.size_y
                    K[k] = alpha
                    k += 1

                    #Point (i,j-1)
                    I[k] = i + j*self.size_y
                    J[k] = i + (j-1)*self.size_y
                    K[k] = alpha
                    k += 1

                    #Point (i,j+1)
                    I[k] = i + j*self.size_y
                    J[k] = i + (j+1)*self.size_y
                    K[k] = alpha
                    k += 1

        self.Mat1 = sparse.coo_matrix((K,(I,J)),shape=(dimension,dimension)).tocsc()

        #Building the second matrix to solve the system (M from Ax_{n+1}=Mx_{n})
        I = np.zeros(size)
        J = np.zeros(size)
        K = np.zeros(size, dtype=np.complex128)

        k = 0
        for i in range(0,self.size_y):
            for j in range(0,self.size_x):
                #Condition aux frontières nulles aux extrémités (en y)
                if i==0 or i==(self.size_y-1):
                    I[k] = i + j*self.size_y
                    J[k] = i + j*self.size_y
                    K[k] = 0
                    k += 1

                #Conditions aux frontières nulles aux extrémités (en x)
                elif j==0 or j==(self.size_x-1):
                    I[k] = i + j*self.size_y
                    J[k] = i + j*self.size_y
                    K[k] = 0
                    k += 1

                #Points à l'intérieur du domaine
                else:
                    #Point central (i,j)
                    I[k] = i + j*self.size_y
                    J[k] = i + j*self.size_y
                    K[k] = 1.0j + 4*alpha + V[i+j*self.size_y]*dt/2
                    k += 1

                    #Point (i-1,j)
                    I[k] = i + j*self.size_y
                    J[k] = (i-1) + j*self.size_y
                    K[k] = -alpha
                    k += 1

                    #Point (i+1,j)
                    I[k] = i + j*self.size_y
                    J[k] = (i+1) + j*self.size_y
                    K[k] = -alpha
                    k += 1

                    #Point (i,j-1)
                    I[k] = i + j*self.size_y
                    J[k] = i + (j-1)*self.size_y
                    K[k] = -alpha
                    k += 1

                    #Point (i,j+1)
                    I[k] = i + j*self.size_y
                    J[k] = i + (j+1)*self.size_y
                    K[k] = -alpha
                    k += 1

        self.Mat2 = sparse.coo_matrix((K,(I,J)),shape=(dimension,dimension)).tocsc()


    def get_prob(self):
        return (abs(self.psi))**2

    def compute_norm(self):
        return np.trapz(np.trapz((self.get_prob()).reshape(self.size_y,self.size_x), self.x).real, self.y).real

    def step(self):
        #Update the state
        self.psi = spsolve(self.Mat1, self.Mat2.dot(self.psi))

        #Update time
        self.t += self.dt
