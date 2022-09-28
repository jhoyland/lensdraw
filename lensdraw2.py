# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 22:41:58 2022

@author: hoyla
"""

import numpy as np
import math
from enum import IntEnum

# The Lensmaker's Forumula for calculating thin lens focal length from curvature and index
def lensmaker(r0,r1,n,n0=1.0):
    lensPower = (n-n0) * (1/r0 - 1/r1) / n0
    return 1/lensPower

# Return the curvature of a plano convex/concave lens with the specified focal length
def planolenscurvature(f,n):
    return f * (n-1)

# Return the curvature of a symmetrical bi convec/concave lens with the specified focal length
def bilenscurvature(f,n):
    return 2 * f * (n-1)

# Return the sagittal height of a circular segment for a specific radius and chord length.
# This is used in drawing the lenses. Requires c <= 2*R
def sagitta(R,c):
    return R - math.sqrt(R**2 - 0.25 * c**2)

# Return a thin lens transfer matrix
def lensMatrix(f):
    return np.matrix( ((1,0),(-1/f,1)) )

# Return a translation transfer matrix
def translationMatrix(L):
    return np.matrix( ((1,L),(0,1)) )

# Return the identity matrix
def identityMatrix():
    return np.matrix( ((1,0),(0,1)) )
    
# Return a plane refraction transfer matrix
def flatInterfaceMatrix(n0,n1):
    return np.matrix( ((1,0),(0,n0/n1)) )

# Return a spherical surface refraction transfer matrix
def curvedInterfaceMatrix(n0,n1,R):
    return np.matrix( ((1,0),((n0-n1)/(n1*R),n0/n1)) )

# Return a flat plate transfer matrix
def thinPlateMatrix(t,n):
    return np.matrix( ((1,t * n),(0,1)) )

''' 
Convenience functions to produce a LaTeX represnetation for matrices and vectors
'''         
        

def matrixToLaTex(mx,precision):
    
    texString = "\\begin{{pmatrix}}\n {0:.{prec}f} & {1:.{prec}f} \\\\ \n {2:.{prec}f} & {3:.{prec}f} \n\\end{{pmatrix}}"
    
    return texString.format(mx[0,0],mx[0,1],mx[1,0],mx[1,1],prec=precision)


def vectorToLaTex(mx,precision):
    
    texString = "\\begin{{pmatrix}}\n {0:.{prec}f} \\\\ \n {1:.{prec}f} \n\\end{{pmatrix}}"
    
    return texString.format(mx[0,0],mx[1,0],prec=precision)

## System matrix functions

# Cardinal points 
# TO DO: Check this

def getCardinalPoints(systemMatrix):
    
    A = systemMatrix[0,0]
    B = systemMatrix[0,1]
    C = systemMatrix[1,0]
    D = systemMatrix[1,1]
    
    # Determinate of system matrix equals the ratio of refractive indices for image and object spaces.
    # Usually this will equal 1.
    
    nratio = A*D - B*C
    
    if C == 0:
        
        Crec = np.inf
        
    else:
        
        Crec = 1 / C
        
    return (D*Crec,-A*Crec,(D-nratio)*Crec,(1-A)*Crec,(D-1)*Crec,(nratio-A)*Crec)

'''
Get image distance for a supplied system matrix. Object distance of zero implies the system matrix already
contains the translation matrix for the object. 
'''

'''
Object distance is measured positive from first element.
'''

def getImageDistance(systemMatrix, objectDistance = 0):
    
    mag = 0
    
    if abs(objectDistance) == np.inf:  ## Object at infinity 
        
        if systemMatrix[1,0] == 0:   ## Matrix element C is zero so image also at infinity
            si = np.inf
            mag = systemMatrix[1,1]
        else:
            ## If the image is not at infinity for an infinite object, the sys matrix times the translation 
            ## matrix to the image will give matrix elementA 0 in the final matrix
            ## this si is -A/C from the original matrix.
            ## Magnification not really defined, but element B in the final matrix converts height to angle
            si = - systemMatrix[0,0] / systemMatrix[1,0]  
            mag = systemMatrix[0,1] + si * systemMatrix[1,1]
            
    else:
        
        # zero object distance is the default (this implies the optics matrix already contains 
        # the transfer matrix for the actual object distance)
        
        systemMatrix = systemMatrix * translationMatrix(objectDistance)
        
        if systemMatrix[1,1] == 0:
            si = np.inf
            mag = systemMatrix[1,0]
        else:
            si = - systemMatrix[0,1] / systemMatrix[1,1]
            mag = systemMatrix[0,0] + si * systemMatrix[1,0]
                
    return si, mag

'''
Get object distance for a supplied system matrix. Image distance of zero implies the system matrix already
contains the translation matrix for the image. 
'''
    
            
def getObjectDistance(systemMatrix, imageDistance = 0):
    
    if abs(imageDistance) == np.inf:
        
        if systemMatrix[1,0] == 0:
            so = - np.inf
        else:
            so = - systemMatrix[1,1] / systemMatrix[1,0]
            
    else:
        
        systemMatrix = translationMatrix(imageDistance) * systemMatrix
                    
        if systemMatrix[0,0] == 0:
            so = -np.inf
        else:
            so = - systemMatrix[0,1] / systemMatrix[0,0]
                
    return so       

## Enumerations

class blockType(IntEnum):
    
    plane = 0
    aperture = 1
    occlusion = 2
    block = 3


class opticalPlane:
    
    # diameter
    # block type
    # x position
    # name
    # label
    
    def __init__(self,name,label=None,x=0):
        
        self.name = name
        self.x = x
        if label is None:
            self.label = self.name
        else:
            self.label = label 
            
        self.blockMethod = blockType.plane
        self.diameter = 0
            
    def isBlocked(self,h):
        
        return [False,h<self.diameter,h>=self.diameter,True][self.blockMethod]
    
    def getMatrix(self):
        
        return identityMatrix()
    
class aperture(opticalPlane):
    
    def __init__(self,name,label=None,x=0,diameter=1):
        
        super.__init__(self,name,label,x)
        
        self.blockMethod = blockType.aperture
        self.diameter = diameter


class lens(aperture):
    
    def __init__(self,name,label=None,x=0,diameter=1,f=25):
        
        super.__init__(self,name,label,x,diameter)
        
        self.f = f
        
        
    












    