# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 22:55:20 2022

@author: jhoyland
"""

import numpy as np

# Return a thin lens transfer matrix
def lensMatrix(f):
    return np.matrix( ((1,0),(-1/f,1)) )

# Return a translation transfer matrix
def translationMatrix(L):
    return np.matrix( ((1,L),(0,1)) )

# The identity matrix
identityMatrix = np.matrix( ((1,0),(0,1)) )
    
# Return a plane refraction transfer matrix
def flatInterfaceMatrix(n0,n1):
    return np.matrix( ((1,0),(0,n0/n1)) )

# Return a spherical surface refraction transfer matrix
def curvedInterfaceMatrix(n0,n1,R):
    return np.matrix( ((1,0),((n0-n1)/(n1*R),n0/n1)) )

# Return a flat plate transfer matrix
def thinPlateMatrix(t,n):
    return np.matrix( ((1,t * n),(0,1)) )

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

element_types = "lens curvedInterface flatInterface thinPlate"







VACUUM_INDEX = 1.00
DEFAULT_LENS_INDEX = 1.5

class opticalPlane:

    def __init__(self,x,d):

        self.x = x
        self.d = d

    def blocked(self,h):
        
        return (abs(h) > abs(0.5 * self.d) ) ^ (self.d < 0)

    def matrix(self,n0,n1):

        return flatInterfaceMatrix(n0,n1)

class thinLens(opticalPlane):

    def __init__(self,x,d,f,n):

        self.R1 = bilenscurvature(f,n)
        self.R2 = -self.R1
        self.n = n
        self.f = f

    def matrix(self,n0,n1):

        if(n0 == n1):

            return lensMatrix(self.f)

        else:

            pass  # Matrix if there is no 




class opticalPlane:
    
    def __init__(self,x,n,d,R):
        
        self.x = x
        self.n = n
        self.d = d # Negative d represents occlusion rather than aperture
        self.R = R 

        # returns true if ray at height h would be blocked by the element
        
    def blocked(self,h):
        
        return (abs(h) > abs(0.5 * self.d) ) ^ (self.d < 0)
    
    def matrix(self,n0 = VACUUM_INDEX,reverse = False):
        
        if reverse:
            return curvedInterfaceMatrix(self.n, n0, -self.R)
        
        return curvedInterfaceMatrix(n0, self.n, self.R)


    
class traceSegment:
    
    def __init__(self,h,a,x,blocked=False,n=VACUUM_INDEX):
        
        self.h = h
        self.a = a
        self.x = x
        self.blocked = blocked 
        self.n = n
        
    def vector(self):
        
        return np.array([[self.h],[self.a]])

    def createNextSegment(plane):

        L = plane.x - self.x
        
        if L < 0:
            return None
        
        translation = translationMatrix(L)
        matrix = plane.matrix(self.n)
        
        new_vector = matrix * translation * self.vector()
        
        h = new_vector[0,0]
        a = new_vector[1,0]
        x = plane.x
        blocked = plane.blocked(h) | self.blocked
        n = plane.n
        
        return traceSegment(h, a, x, blocked, n)  
     
    
    
def trace(h,a,x,planes,n0=VACUUM_INDEX):
    
    traceSegments = [traceSegment(h,a,x,n=n0)]
    
    for plane in planes:
    
        new_segment = traceSegments[-1].createNextTraceSegment(plane)
        if new_segment is not None:
            traceSegments.append(new_segment)
            
    return traceSegments



            
def thinLens(f,x,d,n=DEFAULT_LENS_INDEX,form="bi",flatFirst=True):
    
    if form=='bi':
        
        R1 = bilenscurvature(f, n)
        R2 = -R1
        return lens(R1, R2, x, d, n)
    
    R = planolenscurvature(f, n)
    
    if flatFirst:
        
        return lens(np.inf, R, x, d, n)
    
    return lens(R, np.inf, x, d, n) 

def lens(R1,R2,x,d,n=DEFAULT_LENS_INDEX,t=0,n0=VACUUM_INDEX,n1=None,centered=True):
    
    if n1 is None:
        n1 = n0
        
    dx = 0.5 * t
        
    if not centered:
        x = x + dx
        
    surface1 = opticalPlane(x-dx, n, d, R1)
    surface2 = opticalPlane(x+dx, n1, d, R2)
    
    return surface1,surface2    

def multipletLens(R,x,d,n,t,n0=VACUUM_INDEX,n1=None,centered=True):
    
    if n1 is None:
        n1 = n0
        
    n.append(n1)
    t.insert(0,0)
    
    tt = sum(t)
    
    dx = 0
        
    if centered:
        
        dx = 0.5 * tt
        
    surfaces = [opticalPlane(x+th-dx,nl,d,Rc) for th,nl,Rc in zip(range(len(t)),t,n,R)]
    
    return surfaces

class opticalElement:
    
    def __init__(self):
        pass

# class opticalPlane:
 #   tra
#     def __init__(self):
        
#         self.x = 0
#         self.diameter = np.inf
#         self.clear_aperture = True
#         self.name = "Element"
#         self.element_type = "plane"
#         self.index = VACUUM_INDEX
        
#     def matrix(self,index_before=VACUUM_INDEX):
        
#         return flatInterfaceMatrix(index_before, self.index)
        
# class curvedInterface(opticalPlane):
    
#     def __init__(self):
        
#         super.__init__()
        
#         self.name = "Interface"
#         self.element_type = "flat_interface"
        
#         self.radius = np.inf
        
#     def matrix(self,index_before=VACUUM_INDEX):
        
#         return curvedInterfaceMatrix(index_before, self.index, self.radius)
        
# class lens(opticalPlane):
    
#     def __init__(self):
        
#         super.__init__
        
    
        
        
class plane:

    def __init__(self,name,label=None):

        if label is None:
            self.label = name
        else:
            self.label = label

        self.name = name
        self.x = 0
        self.diameter = np.inf

    def get_matrix(n0,n1):

        return flatInterfaceMatrix(n0,n1)

    def blocked(self,h):
        
        return (abs(h) > abs(0.5 * self.diameter) ) ^ (self.diameter < 0)


class surface:
    
    def __init__(self,name,label=None):
        
        


class surface

