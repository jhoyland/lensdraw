# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 15:36:04 2020

@author: James Hoyland

lensdraw.py

lensdraw is a library for generating to-scale SVG ray diagrams of simple optical systems.

Features:
    
    Create linear arrays of lenses, apertures and other components
    Accurately trace light rays through the lens optics using optical matrices
    Perform matrix calculations on systems of lenses
    Locate stops and pupils
    Output diagrams of optical systems and rays in SVG format

"""

# Here's  acomment


import svgwrite
from svgwrite.extensions import Inkscape
import numpy as np
import math
from enum import IntEnum
from copy import deepcopy

DEFAULT_INDEX = 2.3

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
Get image distance for a supplied system matrix. Object distance of zero implies the system matrix already
contains the translation matrix for the object. 
'''

'''
Object distance is measured positive from first element.
'''

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
        

def matrixToLaTex(mx,precision):
    
    texString = "\\begin{{pmatrix}}\n {0:.{prec}f} & {1:.{prec}f} \\\\ \n {2:.{prec}f} & {3:.{prec}f} \n\\end{{pmatrix}}"
    
    return texString.format(mx[0,0],mx[0,1],mx[1,0],mx[1,1],prec=precision)


def vectorToLaTex(mx,precision):
    
    texString = "\\begin{{pmatrix}}\n {0:.{prec}f} \\\\ \n {1:.{prec}f} \n\\end{{pmatrix}}"
    
    return texString.format(mx[0,0],mx[1,0],prec=precision)


default_plane_style    = {"stroke":"#090909","width":0.5,"fill":"none","stroke_dasharray":"8,8"}
default_lens_style     = {"stroke":"#000000","width":0.5,"fill":"#CCCCCC","stroke_dasharray":"100,0"}
default_aperture_style = {"stroke":"#000000","width":1.5,"fill":"none","stroke_dasharray":"100,0"}
default_trace_style          = {"stroke":"#FF0000","width":0.5,"fill":"none","stroke_dasharray":"100,0"}
default_trace_virtual_style  = {"stroke":"#FF0000","width":0.5,"fill":"none","stroke_dasharray":"1,1"}
default_trace_blocked_style  = {"stroke":"#440000","width":0.5,"fill":"none","stroke_dasharray":"100,0"}


# Class to hold entire optical optics.
# Elements are added to a list which is then sorted by x position of the element

class opticalSystem:

    def __init__(self):

        self.elements = []
        
    # Align
        
    def align(self,alignment = 'first'):
        
        x0 = self.elements[0].x
        
        for el in self.elements:
            el.x = el.x - x0
            
    def getTotalLength(self):
        
        return self.elements[-1].x - self.elements[0].x
         
    # Add a new element and sort it according to its x-position along the optical axis
        
    def addElement(self,el):
        if isinstance(el,opticalPlane):
          el.optics = self
          self.elements.append(el)
          self.elements.sort(key = lambda e : e.x)
          self.align()
          return True
        else:
          return False
      
    # Add a new element a distance x after another element selected by name, to add before use negative x
        
    def addElementAfter(self,name,el,x):
        if isinstance(el,opticalPlane):
            
          parentel = self.getElementByName(name) 
          
          if parentel is not None:
              el.optics = self
              el.x = parentel.x + x
              self.elements.append(el)
              self.elements.sort(key = lambda e : e.x)
              self.align()
              return True
        
        return False
    
    def addImageSensor(self,name,x,sz,afterElement = None):
        
        sensor = imageSensor(name)
        sensor.diameter = sz
        
        if afterElement is not None:
            
            self.addElementAfter(afterElement,sensor,x)
            
        else:
            
            sensor.x = x
            self.addElement(sensor)
    
      
    # Retrieve a specific element according to its name. If the name already exists only the first instance is returned
      
    def addLensFromRadii(self,name,x,r0,r1,d,afterElement=None):
        
        lens = thinLens(name)
        lens.diameter = d
        lens.fFromLensmaker(r0,r1)
        
        if afterElement is not None:
            
            self.addElementAfter(afterElement,lens,x)
            
        else:
            
            lens.x = x
            self.addElement(lens)
            
    def addLens(self,name,x,f,d,form='bi',flatLeft=False,afterElement=None):
        
        lens = thinLens(name)
        lens.diameter = d
        lens.setf(f,form,flatLeft)
                
        if afterElement is not None:
            
            self.addElementAfter(afterElement,lens,x)
            
        else:
            
            lens.x = x
            self.addElement(lens)
            
    def addAperture(self,name,x,d,afterElement=None):
        
        app = aperture(name)
        app.diameter =d
        
        if afterElement is not None:
            
            self.addElementAfter(afterElement,app,x)
            
        else:
            
            app.x = x
            self.addElement(app)
        
        
        
    
    def getElementByName(self,name):
        el = [e for e in self.elements if e.name == name]
        if len(el):
            return el[0]
        else:
            return None


    def getLenses(self):

        return [e for e in self.elements if e.drawType == "lens"]    


    
    
        
    def removeElement(self,name):
        el = self.getElementByName(name)
        
        if el is not None:
            self.elements.remove(el)
        
          
                
    """
    This returns the optics matrix. The default is to find the matrix for every element in the optics. 
    Optionally you can specify a from and to element by name. You can choose to run the stack backwards to get the opposite
    matrix. You can also optionally exclude the starting and ending elements using the inclusive tuple. In this case the
    translation matrix to or from the first or last element is inluded but the not the matrix for the element itself.
    Note: it cannot at the moment automatically work out if it should go forwards or backwards, also if your specified
    elements are not in the optics it may give unexpected results.
    
    TO DO: automatically check for direction
    chck for incorrect elements
    """
            
    def getSystemMatrix(self,fromElement = '__all__', toElement = '__all__',backward = False, inclusive = (True,True)):
        
        sysMatrix = None
        x0 = 0
        
        elementList = (self.elements,self.elements[::-1])[backward]
        direction = (1,-1)[backward]

        
        if toElement == '__all__' or self.getElementByName(toElement) is None:
            lastElement = elementList[-1].name
        else:
            lastElement = toElement
        
        for element in elementList:
            
            if sysMatrix is not None:
                
                sysMatrix = translationMatrix(direction * (element.x - x0)) * sysMatrix
                
                # Skip the last element if we are not including its matrix
                if not (element.name == lastElement and not inclusive[1]):
                    sysMatrix = element.getMatrix() * sysMatrix   
                                    
            else:
                
                if fromElement == element.name or fromElement == '__all__':
                    
                    
                    if inclusive[0]:
                        sysMatrix = element.getMatrix()
                    else:
                        sysMatrix = identityMatrix()

            x0 = element.x
            
            if toElement == element.name:
                
                break
                        
        return sysMatrix
    


class opticalPlane:
    
    
    def __init__(self,name,label=None):
        if label is None:
            self.label = name
        else:
            self.label = label
            
                
                
        self.name = name
        self.x = 0
        self.optics = None
        
        self.physical = False
        
        self.diameter = np.inf
        
        self.clear_aperture = True
        
        self.drawType = "plane"
        
    def __str__(self):
        
        ret = "Optical Plane: {:} @{:} diameter={:}".format(self.name,self.x,self.diameter)
        return ret
        
    # Transfer matrix for ray calculations. Default is the identity matrix
        
    def getMatrix(self):
        return identityMatrix()
    
    def processRay(self,ray):
        return self.getMatrix().dot(ray)
    
    def rayBlocked(self,rayHeight): 
        return self.physical and (self.clear_aperture != (rayHeight < 0.5*self.diameter))
    
    # SVG functions use double dispatch to sparate drawing from calculations
    
    def svgPath(self,drawing):
                   
        return drawing.drawPlane(self.x)
    
    def svgStyle(self,drawing):
        
        return drawing.default_plane_style


# This is for the object being imaged

class physicalObject(opticalPlane):
    
    def __init__(self,name,label=None):
        
        super().__init__(name,label)
        
        self.drawType = "object"
        
        
    def svgPath(self,drawing,scale=1):
           
        return drawing.drawObject(self.x,self.diameter,scale=scale)
    
        

class imagePlane(opticalPlane) :

    def __init__(self,name,label=None):
        super().__init__(name,label)
        
        self.mag = 1
        self.source = None # Refence to the object this is an image of
        
        self.drawType = "object"

    def svgPath(self,drawing):
        
        return self.source.svgPath(drawing,scale=self.mag)
    

class beamBlock(opticalPlane):
    
    def __init__(self,name,label=None):
        super().__init__(name,label)
        self.clear_aperture = False
        self.drawType = "beam_block"
   
class occluder(opticalPlane):
    
    def __init__(self,name,label=None):
        super().__init__(name,label)
        self.clear_aperture = False
        self.diameter = 25.4
        self.drawType = "occluder"
        
    def svgPath(self,drawing):
        
        return drawing.drawOccluder(self.x,self.diameter)

        
                
            
class aperture(opticalPlane):
    
    def __init__(self,name,label=None):
        super().__init__(name,label)
        self.diameter = 25.4
        
        self.physical = True
        self.drawType = "aperture"
        
    # Creates an aperture with the same position and diameter as another object
        
    @classmethod 
    def from_element(cls,el,name,label=None):
        
        ap = cls(name,label)
        ap.diameter = el.diameter
        ap.x = el.x
        return ap
        
        
    def __str__(self):
        
        ret = "Physical aperture: {:} @{:} diameter={:}".format(self.name,self.x,self.diameter)
        return ret
    
    def svgPath(self, drawing):
        
        return drawing.drawAperture(self.x,self.diameter)
    

        
        
    
class imageSensor(occluder):
    
    def __init__(self,name,label=None):
        super().__init__(name,label)
        
        self.drawType = "sensor"
        
                
    def __str__(self):
        
        ret = "Image sensor: {:} @{:} diameter={:}".format(self.name,self.x,self.diameter)
        return ret
            
    
class apertureImage(aperture):
        
    def __init__(self,name,label=None):
        super().__init__(name,label)
        self.physical = False
        
        self.drawType = "aperture_image"
        
    def __str__(self):
        
        ret = "Effective aperture: {:} @{:} diameter={:}".format(self.name,self.x,self.diameter)
        return ret

class thinLens(aperture):
    
    def __init__(self,name,label=None):
        super().__init__(name,label)
        self.n = DEFAULT_INDEX
        self.f = 100
        self.r0 = bilenscurvature(self.f,self.n)
        self.r1 = -self.r0   
        
        self.drawType = "lens"
        
        

                
    def __str__(self):
        
        ret = "Thin lens: {:} @{:} diameter={:} f={:}".format(self.name,self.x,self.diameter,self.f)
        return ret
    
    def constrainDiameter(self):
        
        largestDiameter = 2 * min(abs(self.r0),abs(self.r1))
        
        self.diameter = min(largestDiameter,self.diameter)
    
    def fFromLensmaker(self,r0=None,r1=None,n=None):
        if r0 is not None:
            self.r0 = r0
            
        if r1 is not None:
            self.r1 = r1
            
        if n is not None:
            self.n = n
            
        self.f = lensmaker(self.r0,self.r1,self.n)
        
        self.constrainDiameter()
        
        return self.f
        
    
    def setf(self,f,form='plano',flatLeft=False):
        
        self.f = f

        print("setting f = {:}".format(f))
        print("form = " + form)
        
        if form == 'plano':
            self.makePlano(flatLeft)
        #elif form == 'bi':
        else:
            self.makeBi()

        print("radii = {:}, {:}".format(self.r0,self.r1))
            
        self.constrainDiameter()        
            
    def getMatrix(self):
        return lensMatrix(self.f)
    
    def swapSurfaces(self):
        r1 = self.r0
        r0 = self.r1
        self.r0 = -r0
        self.r1 = -r1
        
    def makePlano(self,flatLeft=False):
        if flatLeft:
            self.r0 = np.inf
            self.r1 = -planolenscurvature(self.f,self.n)
        else:
            self.r1 = np.inf
            self.r0 = planolenscurvature(self.f,self.n)
            
        self.constrainDiameter()
            
    def makeBi(self):
        r = bilenscurvature(self.f,self.n)
        self.r0 =  r
        self.r1 = -r
            
        self.constrainDiameter()
        
    def svgPath(self,drawing):
        
        return drawing.drawThinLens(self.x,self.diameter,self.r0,self.r1)

        
            

# raySegment: one segment of ray trace between two optical elements. Really just a wrapper for the vector   

class raySegment:

    def __init__(self,x,vector: np.array):
        self.x = x
        self.vector = vector
        self.blocked = False
        self.excluded = False
        self.source = None
        
    def height(self):
        return self.vector[0,0]
    
    def angle(self):
        return self.vector[1,0]
    
    # returns a scaled tuple of the vertex of the ray segment beginning - used for drawing
    
    def xy(self,scale=(1,1)):
        return (float(self.x * scale[0]),float(self.vector[0,0] * scale[1]))
    
    
        

# rayTrace: an individual trace of a ray through the optical system

class rayTrace:
    
    def __init__(self,x,h,angle,group=0):
        
        self.group = group  # Traces can be grouped for drawing purposes
        self.h = h             
        self.angle = angle
        self.x = x
        self.trace = []
        self.optics = None
        
        self.style = deepcopy(default_trace_style)
        self.style_blocked = deepcopy(default_trace_blocked_style)
        self.style_virtual = deepcopy(default_trace_virtual_style)


    def __str__(self):

        ret = 'rayTrace:g{:}, h{:}, a{:}, x{:}, trace_segs:{:}'.format(self.group,self.h,self.angle,self.x,len(self.trace))

        return ret
        
    # Traces the ray through the optical system. Clipping used especially for distanct objects
        
    def propagateRay(self,optics,ray,endOnBlocked = True,startPlane = "__all__", endPlane = "__all__"):
        
        raySegments = [ray]
        
        started = startPlane == "__all__"
        
        i=0
        
        for plane in optics.elements:
            L = plane.x - raySegments[-1].x
            if L > 0:
                newVector = translationMatrix(L) * raySegments[-1].vector
                
                newVector = plane.getMatrix() * newVector
                newSegment = raySegment(plane.x,newVector)
                newSegment.source = plane
                
                    
                newSegment.blocked = raySegments[-1].blocked or plane.rayBlocked(abs(newVector[0,0]))
                    
                raySegments.append(newSegment)
        
                
                if not started:
                    i = i+1
                    if startPlane == plane.name:
                        started = True
                
                if endOnBlocked and newSegment.blocked:
                    break
                
                if plane.name == endPlane:
                    break
                         
        return raySegments[i:]
    
    
    def propagateThrough(self,optics,clipStart = 0, clipEnd = None, toBlocked = True, startPlane = "__all__", endPlane = "__all__"):
        
        dx = 0
        
        if self.x >= clipStart:
            
            x0 = self.x
            
        else:
                
            x0 = clipStart
            
            if self.x == -np.inf:
                
                dx = clipStart
                
            else:
                
                dx = clipStart - self.x
                                    
        dh = dx * self.angle
        
        startRay = raySegment(x0,np.array([[self.h+dh],[self.angle]]))
        self.trace.clear()
        self.trace = self.propagateRay(optics,startRay,toBlocked,startPlane,endPlane)
        
        
        
        if clipEnd is not None:
            if not(toBlocked and self.trace[-1].blocked):
                L = clipEnd - self.trace[-1].x
                if L > 0:
                    newVector = translationMatrix(L) * self.trace[-1].vector
                    newElement = raySegment(clipEnd,newVector)
                    newElement.blocked = self.trace[-1].blocked
                    self.trace.append(newElement)
                
        self.optics = optics
                
                

        
'''
tracingProject:
    collects together the optics along with a specific object and associated tracing and calculation tasks.
    it will use the optics matrix to calculate image distances,
'''



class tracingProject:
    
    DEFAULT_OUTPUT_LOCATION = 100
    DEFAULT_INPUT_LOCATION = -25.4
    
    def __init__(self,optics):
        
        self.optics = optics
        self.object = None
        self.image = None
        self.intermediateImages = []
        self.traces = []
        # self.logicalApertures["entrance_pupil"] = None
        # self.logicalApertures["exit_pupil"] = None
        # self.logicalApertures["aperture_stop"] = None
        # self.logicalApertures["field_stop"] = None
        # self.logicalApertures["exit_window"] = None
        # self.logicalApertures["entrance_window"] = None

        self.logicalApertures = {
            "entrance_pupil":None,
            "exit_pupil":None,
            "aperture_stop":None,
            "field_stop":None,
            "entrance_window":None,
            "exit_window":None
        }
        
        self.valid = False
        
        self.angularAperture = 0
        self.FOV = math.pi
        
        self.inputLocation = 0
        self.outputLocation = 0
        
        self.traceGroup = 0
        
    # SETUP
    
    def setObject(self, x, size = 25.4):
        
        self.valid = False
                
        self.object = physicalObject('object')
        
        self.object.x = x
        self.object.diameter = size
        
        self.object.optics = self.optics

    def setInputPlane(self,position=DEFAULT_INPUT_LOCATION,relativeTo='object'):
        
        if relativeTo=='object':
            
            if self.object is None:
                
                relativeTo = 'firstElement'
                
            else:
                
                if abs(self.object.x) == np.inf:
                    
                    self.inputLocation = position
                    
                else:
                    
                    self.inputLocation = self.object.x + position
                    
        
        if relativeTo=='firstElement':
            
            self.inputLocation = position
        
    def setOutputPlane(self,position=DEFAULT_OUTPUT_LOCATION,relativeTo='auto'):
        
        if relativeTo == 'auto':
            
            if self.imageIsVirtual():
                
                relativeTo = 'lastElement'
                
            else:
                
                relativeTo = 'image'
                position = 0
                    
        
        if relativeTo =='image':
            
            if self.image is None:
                
                relativeTo = 'lastElement'
                
            else:
                
                L = self.image.x + position
                
                if L < self.optics.getTotalLength() or abs(L) == np.inf:
                    
                    relativeTo = 'lastElement'
                    
                else:
                    
                    self.outputLocation = L
                    
        if relativeTo == 'lastElement':
            
            self.outputLocation = position + self.optics.getTotalLength()
            
        if relativeTo == 'firstElement':
            
            self.outputLocation = position
           
# SOLVE SYSTEM
      
        
    def calculateImage(self):
        
        if self.object is not None:
            
            so = - self.object.x
            
            si, m = getImageDistance(self.optics.getSystemMatrix(),so)
            
            self.image = imagePlane('image')

            self.image.source = self.object
            self.image.mag = m
            
            if abs(so) != np.inf and abs(si) != np.inf:
                self.image.mag = m
                
            self.image.diameter = self.object.diameter * m
            
            self.image.x = si + self.optics.elements[-1].x
            
            self.image.optics = self.optics
            
            self.imageDistance = si
            
            return True
        
        
        return False
    
    def getImageDistance(self):
        
        if self.image is None:
            
            self.calculateImage()
            
        if self.image is not None:
            
            return self.imageDistance
        
        else:
            
            return None
            
    def imageIsVirtual(self):
        
        if self.image is not None:
            
            return (self.image.x - self.optics.elements[-1].x < 0) or (self.image.x == np.inf)
            
        return False
    
                
            
    
    """
    To find the aperture stop. Calculate the optics matrix up to each successive plane. For objects not at infinity
    include the translation matrix from the object plane to the first element.  For each plane calculate the input angle (ai)
    from the aperture diameter (d) at the plane and the partial matrix as
    
    ai = d / (2*B)
    
    The element producing the smallest angle is the aperture stop.
    If element B of the partial matrix is zero then the plane is conjugate with the object plane and cannot be the 
    aperture stop (though it can be the field stop)
    
    If the object is at infinity the same procedure is done starting with the first physical element of the optics. This time
    the input ray height parallel to the axis (hi) is calculated is
    
    hi = d / (2*A)
    
    Again if element A is zero there is an intermediate image at the plane and so it is not the aperture stop.
    """
    
    def clearLogicalApertures():

        self.logicalApertures = {x:None for x in self.logicalApertures}
    
    def findApertureStop(self):
        
        self.angularAperture = np.inf
        
        finiteObject = abs(self.object.x) != np.inf


        print("Finding Aperture  stop")
        
            
        
        initialTransferMatrix = (identityMatrix(),translationMatrix(abs(self.object.x)))[finiteObject]
        
        for element in self.optics.elements:
            print("Checking element: " + element.name)
            
            if element.diameter < np.inf:
                
                syMx = self.optics.getSystemMatrix(toElement = element.name) * initialTransferMatrix
                mxel = syMx[0,(0,1)[finiteObject]] # Select element A or B
                
          #      print(syMx)
         #       print(mxel)
                
                if mxel != 0:
                
                    r0 = abs(element.diameter / (2*mxel))
                
                    if r0 < self.angularAperture:
                        self.logicalApertures["aperture_stop"] = element
                        self.angularAperture = r0

            
    def findEntrancePupil(self):
        
        if self.logicalApertures["aperture_stop"] is not None:
            
            self.logicalApertures["entrance_pupil"] = apertureImage('__entrance_pupil__')
            self.logicalApertures["entrance_pupil"].optics = self.optics
            self.logicalApertures["entrance_pupil"].drawType = "entrance_pupil"
            firstElement = self.optics.elements[0]
            
            if self.logicalApertures["aperture_stop"].name == firstElement.name:
                          
                self.logicalApertures["entrance_pupil"].x = firstElement.x
                self.logicalApertures["entrance_pupil"].diameter =  firstElement.diameter
                
            else:
            
                syMx = self.optics.getSystemMatrix(fromElement = self.logicalApertures["aperture_stop"].name ,backward = True,inclusive = (False,True))
                
                si,mag = getImageDistance(syMx)
                
      #          print("Entrance pupil image distance {:}".format(si))
                            
                self.logicalApertures["entrance_pupil"].x = firstElement.x - si
                self.logicalApertures["entrance_pupil"].diameter =  abs(self.logicalApertures["aperture_stop"].diameter * (syMx[0,0] + si * syMx[1,0]))
            
            return True
        
        return False
    
    
    def findExitPupil(self):
                        
        if self.logicalApertures["aperture_stop"]  is not None:
            
            self.logicalApertures["exit_pupil"] = apertureImage('__exit_pupil__')
            self.logicalApertures["exit_pupil"].optics = self.optics
            self.logicalApertures["exit_pupil"].drawType = "exit_pupil"
            lastElement = self.optics.elements[-1]
            
            if self.logicalApertures["aperture_stop"].name == lastElement.name:
                
                self.logicalApertures["exit_pupil"].x = lastElement.x          
                self.logicalApertures["exit_pupil"].diameter = lastElement.diameter
            
            else:
            
                syMx = self.optics.getSystemMatrix(fromElement = self.logicalApertures["aperture_stop"].name,inclusive = (False,True))
                
                si,mag = getImageDistance(syMx)
                
                self.logicalApertures["exit_pupil"].x = lastElement.x + si            
                self.logicalApertures["exit_pupil"].diameter = abs(self.logicalApertures["aperture_stop"].diameter * (syMx[0,0] + si * syMx[1,0]))
            
            return True
        
        return False
    
                       
    def findFieldStop(self):
        
        aF = np.inf
        
       # finiteObject = abs(self.object.x) != np.inf
        
        #initialTransferMatrix = (identityMatrix(),translationMatrix(abs(self.object.x)))[finiteObject]
        
        print("Finding field stop")

        if len(self.optics.elements) <= 1:

            self.logicalApertures["field_stop"] = None

            return
        
        for element in self.optics.elements:
            
            print("Checking element: " + element.name)
            
            if element.diameter < np.inf:


                
                if not(element.name == self.logicalApertures["aperture_stop"].name):
                
                    direction = element.x < self.logicalApertures["aperture_stop"].x
                    
   #                 if direction:
    #                    print("Before AS")
     #               else:
      #                  print("After AS")
                    
                    syMx = self.optics.getSystemMatrix(fromElement = self.logicalApertures["aperture_stop"].name, toElement = element.name,backward = direction, inclusive = (False,False))
                       
   #                 print("System matrix: {:}".format(syMx))
                    
                    mxel = syMx[0,1] 
                    
   #                 print("Element B = {:}".format(mxel))
                
                    if mxel != 0:
                                            
                        r0 = abs(element.diameter / (2*mxel)) % (2*math.pi)
                        
                        print("Diameter = {:}   Angle = {:}".format(element.diameter,r0))
                    
                        if r0 < aF:
                            
     #                       print("New Field Stop")
                            self.logicalApertures["field_stop"] = element
                            aF = r0
           
            
                
    def findEntranceWindow(self):
        
        if self.logicalApertures["field_stop"] is not None:
            
            self.logicalApertures["entrance_window"] = apertureImage('__entrance_window__')
            self.logicalApertures["entrance_window"].optics = self.optics
            self.logicalApertures["entrance_window"].drawType = "entrance_window"
            firstElement = self.optics.elements[0]
            
            if self.logicalApertures["field_stop"].name == firstElement.name:
                          
                self.logicalApertures["entrance_window"].x = firstElement.x
                self.logicalApertures["entrance_window"].diameter =  firstElement.diameter
                
            else:
            
                syMx = self.optics.getSystemMatrix(fromElement = self.logicalApertures["field_stop"].name ,backward = True,inclusive = (False,True))
                
                si,mag = getImageDistance(syMx)
                
        #        print("Entrance window image distance {:}".format(si))
                            
                self.logicalApertures["entrance_window"].x = firstElement.x - si
                
                if abs(si) == np.inf:
                    # angular diameter for infinite entrance window distance
                    self.logicalApertures["entrance_window"].diameter =  abs(self.logicalApertures["field_stop"].diameter * syMx[1,0])  
                else:
                    self.logicalApertures["entrance_window"].diameter =  abs(self.logicalApertures["field_stop"].diameter * (syMx[0,0] + si * syMx[1,0]))
            
            return True

        print("No entranceWindow")
        
        return False   
    
                    
    def findExitWindow(self):
        
        if self.logicalApertures["field_stop"] is not None:
            
            self.logicalApertures["exit_window"] = apertureImage('__exit_window__')
            self.logicalApertures["exit_window"].optics = self.optics
            self.logicalApertures["exit_window"].drawType = "exit_window"
            lastElement = self.optics.elements[-1]
            
            if self.logicalApertures["field_stop"].name == lastElement.name:
                          
                self.logicalApertures["exit_window"].x = lastElement.x
                self.logicalApertures["exit_window"].diameter =  lastElement.diameter
                
            else:
            
                syMx = self.optics.getSystemMatrix(fromElement = self.logicalApertures["field_stop"].name,inclusive = (False,True))
                
                si,mag = getImageDistance(syMx)
                
        #        print("Entrance window image distance {:}".format(si))
                            
                self.logicalApertures["exit_window"].x = si + lastElement.x
                
                if abs(si) == np.inf:
                    # angular diameter for infinite exit window distance
                    self.logicalApertures["exit_window"].diameter =  abs(self.logicalApertures["field_stop"].diameter * syMx[1,0])  
                else:
                    self.logicalApertures["exit_window"].diameter =  abs(self.logicalApertures["field_stop"].diameter * (syMx[0,0] + si * syMx[1,0]))
            
            return True
        
        return False   



    def getFOV(self):

        if self.logicalApertures["field_stop"] is None:

            self.FOV = math.pi * 0.999

            return

        print(self.logicalApertures["entrance_window"])
        
        dEW = 0.5* self.logicalApertures["entrance_window"].diameter
        #dEP = 0.5* self.logicalApertures["entrance_pupil"].diameter

        xEW = self.logicalApertures["entrance_window"].x
        xEP = self.logicalApertures["entrance_pupil"].x
        
        if abs(xEW) == np.inf:
            
            ao = dEW
            
        else:
        
            dx = abs(xEW - xEP)
            ao = dEW / dx
        
        self.FOV = ao * 2

    def solveAll(self):
        
        self.calculateImage()
        self.findApertureStop()
        self.findExitPupil()
        self.findEntrancePupil()
        self.findFieldStop()
        self.findEntranceWindow()
        self.findExitWindow()
        self.getFOV()
        


# TRACING
 
    def addTrace(self,h,a,group=0):
        
        if h == 'top':
            h = self.object.diameter * 0.5
        if h == 'bottom':
            h = -self.object.diameter * 0.5
            
        self.traces.append(rayTrace(self.object.x,h,a,group))
        
    def addTracesAngleRange(self,h,angles,group=0):
        
        for angle in angles:                      
            self.addTrace(h,angle,group)
                
    def addTracesHeightRange(self,hs,a,group=0):
        
        for h in hs:                      
            self.addTrace(h,a,group)


            
    def addTracesFillFirstElement(self,h,fill_factor=0.9,numberStep=10,method='number',group=0):
        
                
        if h == 'top':
            h = self.object.diameter * 0.5
        if h == 'bottom':
            h = -self.object.diameter * 0.5
        
        d = fill_factor * self.optics.elements[0].diameter * 0.5
        dx = - self.object.x
        
        amin = - (d+h) / dx
        amax = (d-h) / dx
        
        if method == 'step':
            
            numberStep =math.ceil((amax - amin) / numberStep)
 
        
        angles = np.linspace(amin,amax,numberStep )
        
        self.addTracesAngleRange(h, angles, group)
        
        
    def addMarginalRay(self,group=0,negative=False):
        
                
        if self.logicalApertures["aperture_stop"] is not None:
            
            aa = 0.99*self.angularAperture
            
            if negative:
                
                aa = -aa
            
            if abs(self.object.x) == np.inf:
                
                self.addTrace(aa,0,group)
                
            else:
                
                self.addTrace(0,aa,group)  

    # Adds a chief ray based on height. For a finite object the height is the height off the axis at the object point
    # For an object at infinity the height is the height the ray at the first element


    def addChiefRayInfiniteObject(self,a=0,method='angle',group=0,rays=1):

        if self.logicalApertures["entrance_pupil"] is not None:

            if method == "top":

                method = 'FOV'
                a = 1

            if method == "bottom":

                method = 'FOV'
                a = -1

            if method in "FOVobject":

                a = a * self.FOV * 0.5 * 0.99

            h = -a * (self.optics.elements[0].x - self.logicalApertures["entrance_pupil"].x)

            if rays <= 1:

                hh = [h]

            else:

                ap = self.angularAperture * 0.999
                hh = np.linspace(h - ap, h + ap, rays)

            self.addTracesHeightRange(hh,a,group)            


    def addChiefRayFiniteObject(self,h=0,method='height',group=0,rays=1):

        if self.logicalApertures["entrance_pupil"] is not None:

            d = (self.object.x - self.logicalApertures["entrance_pupil"].x)

            print("h argument = " + str(h))

            if method=="top":    # Rays from top of object

                method="object"
                h=1

            if method=="bottom": # Rays from bottom of object

                method="object"
                h=-1

            if method=="object": # Rays from height relative to object height

                h = h * self.object.diameter * 0.5

            if method=="FOV":   # Rays from height relative to field of view

                a = h * self.FOV * 0.5 * 0.99
                h = a * d

            else: # For both 'object' and 'height' angle is calulated from h

                a = h / d 


            if rays <= 1:

                aa = [a]

            else:

                ap = self.angularAperture * 0.999
                aa = np.linspace(a - ap, a + ap, rays)

            self.addTracesAngleRange(h,aa,group)  

    def addChiefRays(self,h=1,method="FOV",group=0,rays=1):

        if self.object.x == -np.inf:

            self.addChiefRayInfiniteObject(a=h,method=method,group=group,rays=rays)

        else:

            self.addChiefRayFiniteObject(h=h,method=method,group=group,rays=rays)


    
    def traceAll(self,toBlocked = True, clip = True):
        
        for trace in self.traces:
            
            if clip:
                c = self.outputLocation
            else:
                c = None

            trace.propagateThrough(self.optics,clipStart = self.inputLocation, clipEnd = c, toBlocked = toBlocked)
       

# DRAWING 

    def report(self):
        
        print("Optical system:")
        
        for e in self.optics.elements:
            
            print(e)
            
        print("Object @{:}".format(self.object.x))
        print("Final Image @{:} (From last element = {:})".format(self.image.x,self.imageDistance))

        if self.imageIsVirtual():

            print("Virtual image")

        else:

            print("Real image")

        
        print("Aperture stop = {:}".format(self.logicalApertures["aperture_stop"].name))

        if self.logicalApertures["field_stop"] is not None:
            print("Field stop = {:}".format(self.logicalApertures["field_stop"].name))
        
        degFOV = math.degrees(self.FOV)
        
        print("Field of View: {:}".format(degFOV))
        
        print(self.logicalApertures["entrance_pupil"])
        print(self.logicalApertures["exit_pupil"])
        print(self.logicalApertures["entrance_window"])
        print(self.logicalApertures["exit_window"])
              
    

              
            
class lensrender:
    
    DEFAULT_SCALE_POSITIONING = 3  # 1 pixel = 1 mm, used for positioning of planes
    DEFAULT_SCALE_ELEMENTS = 5  # 1 pixel = 1 mm, used for visual sizing of lenses and other elements
    DEFAULT_SCALE_CURVATURE = 3 # Exaggerate or reduce visual curvature of lenses

    DEFAULT_DISPLAY_HEIGHT = 140
    DEFAULT_DISPLAY_WIDTH = 960
    
    DEFAULT_ELEMENT_THICKNESS = 2
    
 
    
    
    
    def __init__(self,project=None,name="untitled.svg",size=(DEFAULT_DISPLAY_WIDTH,DEFAULT_DISPLAY_HEIGHT),profile="full"):
    
        self.display_height = size[1]
        self.display_width = size[0]
        self.scale_position = lensrender.DEFAULT_SCALE_POSITIONING 
        self.scale_elements = lensrender.DEFAULT_SCALE_ELEMENTS 
        self.scale_curvature = lensrender.DEFAULT_SCALE_CURVATURE 
        self.element_thickness = lensrender.DEFAULT_ELEMENT_THICKNESS 
        self.axis_height = 0.5 * self.display_height
        self.x_origin = -30
        self.max_Aperture = 100
        self.project = project
        
        self.svgdrawing = svgwrite.Drawing(name,size=size,profile=profile)
        self.inkscape = Inkscape(self.svgdrawing)
        
        # self.axisLayer = self.inkscape.layer(label="axis")
        # self.opticLayer = self.inkscape.layer(label="optics")
        # self.stopsLayer = self.inkscape.layer(label="stops")
        # self.pupilsLayer = self.inkscape.layer(label="pupils")
        # self.windowsLayer = self.inkscape.layer(label="windows")
        
        # self.svgdrawing.add(self.opticLayer)
        
        self.rayTraceLayers = {}
        
        self.layers = {
            "axis":self.inkscape.layer(label="axis"),
            "optics":self.inkscape.layer(label="optics"),
            "stops":self.inkscape.layer(label="stops"),
            "pupils":self.inkscape.layer(label="pupils"),
            "windows":self.inkscape.layer(label="windows"),  
            "objects":self.inkscape.layer(label="objects"),
            "images":self.inkscape.layer(label="images")  
        }
        
        for layer in self.layers:
            self.svgdrawing.add(self.layers[layer])
            
        self.group_colors = ["#FF0000","#E00000","#29ab93","#0000FF","#00FFFF","#FF00FF","FFFF00"]
        
        self.styles = {    
        "missing_style":{"stroke":"#FF0000","stroke_width":3,"fill":"none","stroke_dasharray":"3,3"},
        "plane":{"stroke":"#090909","stroke_width":0.5,"fill":"none","stroke_dasharray":"8,8"},
        "lens":{"stroke":"#000000","stroke_width":1.0,"fill":"#CCCCCC","stroke_dasharray":"100,0"},
        "aperture":{"stroke":"#000000","stroke_width":1.5,"fill":"none","stroke_dasharray":"100,0"},
        "trace":{"stroke":"#FF0000","stroke_width":0.5,"fill":"none","stroke_dasharray":"100,0"},
        "virtual_trace":{"stroke":"#FF0000","stroke_width":0.5,"fill":"none","stroke_dasharray":"1,1"},
        "blocked_trace":{"stroke":"#440000","stroke_width":0.5,"fill":"none","stroke_dasharray":"100,0"},
        "aperture_stop":{"stroke":"#009900","stroke_width":1.5,"fill":"none","stroke_dasharray":"100,0"},
        "field_stop":{"stroke":"#990000","stroke_width":1.5,"fill":"none","stroke_dasharray":"100,0"},
        "entrance_pupil":{"stroke":"#007710","stroke_width":1.5,"fill":"none","stroke_dasharray":"100,0"},
        "exit_pupil":{"stroke":"#005010","stroke_width":1.5,"fill":"none","stroke_dasharray":"100,0"},
        "entrance_window":{"stroke":"#770010","stroke_width":1.5,"fill":"none","stroke_dasharray":"100,0"},
        "exit_window":{"stroke":"#500010","stroke_width":1.5,"fill":"none","stroke_dasharray":"100,0"},
        "axis_style":{"stroke":"#040404","stroke_width":2.0,"fill":"none","stroke_dasharray":"10,4"},
        "object_style":{"stroke":"#000000","stroke_width":1.0,"fill":"#000000","stroke_dasharray":"100,0"},
        "image_style":{"stroke":"#020202","stroke_width":1.0,"fill":"#020202","stroke_dasharray":"100,0"}
    
        }

        self.path_method = {
            "lens":self.svgThinLens,
            "aperture":self.svgAperture,
            "object":self.svgObject,
            "plane":self.svgPlane
        }
        
    def save_drawing(self):
        
        self.svgdrawing.save()
        
        
    ## Builds the SVG path text for a lens of given radii and diameter
    
    def get_lens_surface_params(self,r,d):
        
        if abs(r) == np.inf:
            
            return r,0
        
        R = r * self.scale_curvature
        D = d * self.scale_elements
        
        if abs(R) < 0.5*D:
            
            R = math.copysign(0.5*D, R)
            
        sag = math.copysign(sagitta(abs(R),D),R)
        
        return R,sag
    
    def svgPlane(self,el):
        
        sx, sy = self.scale_element_position(el)
        
        path = svg_path(sx,sy)
        
        path.moveRel(0, -0.5*self.display_height)
        path.verticalRel(self.display_height)
    
        return path

    def svgObject(self,el):

        sx, sy = self.scale_element_position(el)

        path = svg_path(sx,sy)

        d = math.copysign(1,el.diameter) # Makes sure arrowhead points right way (-ve diameter for inverted image)

        path.moveRel(0, 0.5 * el.diameter * self.scale_elements)
        path.verticalRel(-1 * el.diameter * self.scale_elements)
        path.lineRel(-3,7*d)
        path.lineRel(6,0)
        path.lineRel(-3,-7*d)

        return path
    
    def svgAperture(self,el):
        
        sx, sy = self.scale_element_position(el)
        
        path = svg_path(sx,sy)
        
        disp_diameter = el.diameter * self.scale_elements
        h = self.max_Aperture * self.scale_elements
            
        if disp_diameter > h-5:
            
            h = 5
            
        else:
            
            h = 0.5 * (h - disp_diameter)
               
        path.moveRel(-0.5 * self.element_thickness, 0.5 * disp_diameter)
        path.horizontalRel(self.element_thickness)
        path.moveRel(0,-disp_diameter)
        path.horizontalRel(-self.element_thickness)
        
        path.moveRel(0.5 * self.element_thickness, 0)
        path.verticalRel(-h)
        path.moveRel(0,h + disp_diameter)
        path.verticalRel(h)
        
        return path
    
    def svgThinLens(self,el):
        
        sx, sy = self.scale_element_position(el)
        
        path = svg_path(sx,sy)
        
        DLens = el.diameter * self.scale_elements
        
        R0,lsag = self.get_lens_surface_params( el.r0, el.diameter)       
        R1,rsag = self.get_lens_surface_params( el.r1, el.diameter)

        rsag = -rsag # Sagitta for right surface needs to be flipped due to radius sign convention

        print("Drawing lens {:}".format(el.name))

        print("R0={:} lsag={:}".format(R0,lsag))
        print("R1={:} rsag={:}".format(R1,rsag))
  
        # Checks for visual interference between the curved surfaces and pad out thickness if necessary    
            
        thickness = self.element_thickness
            
        rimThickness = thickness
            
        centerThickness = lsag + rsag + thickness
        
        if centerThickness < thickness:
            rimThickness += (rimThickness - centerThickness)
        
        # Move to bottom left corner of lens
        
        path.moveRel(-0.5*rimThickness, -0.5*DLens)
        
        # Draw left face
        
        if lsag == 0: # Plane face
            path.verticalRel(DLens)
        else:         # Curved face
            path.circularArcRel(0,DLens,abs(R0),long_arc = False,clockwise = R0 < 0)
            
        # Draw top edge 
             
        path.horizontalRel(rimThickness)
        
        # Draw right face
        
        if rsag == 0:
            path.verticalRel(-DLens)
        else:
            path.circularArcRel(0,-DLens,abs(R1),long_arc = False,clockwise = R1 > 0)
            
        # Close the path    

        
        path.close()
        
        
        return path
    
## Scaling functions

    def scale_element_position(self,el):

        return self.scale_point((el.x,0))
    
    def scale_point(self,p):
        
        return (p[0] * self.scale_position - self.x_origin, self.axis_height - p[1] * self.scale_elements)

    def scale_length(self,length):

        return length * self.scale_position

    def setHorizontalScaleFromProject(self):

        projectWidth = self.project.outputLocation - self.project.inputLocation
        self.scale_position = self.display_width / projectWidth
        self.x_origin = self.project.inputLocation * self.scale_position

    def setHorizontalScale(self,left,right):

        projectWidth = right - left
        self.scale_position = self.display_width / projectWidth
        self.x_origin = left * self.scale_position


    def setVerticalScale(self,height,fill=1):

        self.scale_elements = fill * self.display_height / height

## SVG Element creatiom

    def renderElement(self,el,layer,style_name=None,name=None,force_type=None):

        scaled_x, scaled_d = self.scale_point((el.x,el.diameter))
        if (abs(scaled_x) < 2*self.display_width) and (abs(scaled_d) < 2*self.display_height):

            if force_type is None:
                func_name = el.drawType
            else:
                func_name = force_type

            path = self.path_method[func_name](el)

            path_string = path.get_path_string()

            if style_name is None:
                
                style_name = el.drawType
            
            if style_name in self.styles: 
                style = self.styles[style_name]
            else:
                style = self.styles["missing_style"]

            path_element = self.svgdrawing.path(path_string,**style)
            path_element.attribs["id"] = el.name

            print(layer)
            self.layers[layer].add(path_element)

    def drawObject(self):

        if self.project.object is not None:

            if abs(self.project.object.x) != np.inf:

                path = self.svgObject(self.project.object)

                path_string = path.get_path_string()

                style = self.styles["object_style"]

                path_element = self.svgdrawing.path(path_string,**style)

                self.layers["objects"].add(path_element)

    def drawImage(self):

        if self.project.image is not None:

            if abs(self.project.image.x) != np.inf:

                path = self.svgObject(self.project.image)

                path_string = path.get_path_string()

                style = self.styles["image_style"]

                path_element = self.svgdrawing.path(path_string,**style)
                self.layers["images"].add(path_element)
        
        
    def drawOpticalSystem(self):
        
        for el in self.project.optics.elements:
            
            #self.drawOpticElement(el, "optics")

            self.renderElement(el,"optics")

    def drawLogicalApertures(self):

        for k,v in self.project.logicalApertures.items():

            if v is not None:

                layer_name = k.split('_')[-1] + 's'

                self.renderElement(v,layer=layer_name,style_name=k,force_type="aperture")

            
    def drawOpticalAxis(self):
        
        axis = self.svgdrawing.line( start = (0,self.axis_height),end = (self.display_width,self.axis_height),** self.styles["axis_style"])
        self.layers["axis"].add(axis)

    def drawAllTraces(self,virtuals=False):

        for ray in self.project.traces:

            self.drawRayTrace(ray,virtuals)
             

    def drawRayTrace(self,ray,virtuals=False):
        
       

        if len(ray.trace) > 0:
        
            if not ray.group in self.rayTraceLayers:

                self.rayTraceLayers[ray.group] = self.inkscape.layer(label = "rayGroup{}".format(ray.group))
                self.svgdrawing.add(self.rayTraceLayers[ray.group]) 
           
            previous = ray.trace[0]

            for element in ray.trace[1:]:
                                
                x1, y1 = self.scale_point(previous.xy())
                x2, y2 = self.scale_point(element.xy())
                
                rstyle =  deepcopy((self.styles["trace"],self.styles["blocked_trace"])[int(previous.blocked)])    
                rstyle["stroke"] = self.group_colors[ray.group] 
                
                line = self.svgdrawing.line( start = (x1,y1),end = (x2,y2),**rstyle)
                self.rayTraceLayers[ray.group].add(line)
            
                previous = element

            if virtuals:

                last_lens = self.project.optics.getLenses()[-1]

                final_ray = [r for r in ray.trace if r.source == last_lens][0]

                x1, y1 = self.scale_point(final_ray.xy())

                xi = self.project.image.x

                if xi < x1:

                    yi = final_ray.height() + (xi - final_ray.x) * final_ray.angle()  # NOT WORKING!!!

                    x2, y2 = self.scale_point((xi,yi))
                
                    rstyle =  deepcopy(self.styles["virtual_trace"])    
                    rstyle["stroke"] = self.group_colors[ray.group] 
                    
                    line = self.svgdrawing.line( start = (x1,y1),end = (x2,y2),**rstyle)
                    self.rayTraceLayers[ray.group].add(line)



    def drawAll(self,virtuals=False):

        self.drawOpticalSystem()
        self.drawOpticalAxis()
        self.drawAllTraces(virtuals)
        self.drawLogicalApertures()
        self.drawObject()
        self.drawImage()
            

class svg_path:
    
    move_relative = "m {x:4},{y:4} "
    line_relative = "l {x:4},{y:4} "
    move_absolute = "M {x:4},{y:4} "
    line_absolute = "L {x:4},{y:4} "
    
    horiz_relative = "h {x:4} "
    verti_relative = "v {y:4} "
    
    arc_relative = "a {r1:4},{r2:4} {xaxis:4} {large_arc} {sweep} {x:4},{y:4} "
    circ_arc_relative = "a {r:4},{r:4} 0 {large_arc} {sweep} {x:4},{y:4} "
    
    
    def __init__(self,x=None,y=0):
        
        
        self.path_text = []
        
        if x is not None:
            
            self.moveAbs(x,y)
        
        self.path_color = "black"
        self.path_dash = None
        self.fill = False
        self.fill_color = "black"
        self.path_width = 1
        
        
    def appendSection(self,s0,**kwargs):
        
        s1 = s0.format(**kwargs)

        c = self.unclose()
        self.path_text.append(s1)
        
        if c:
            
            self.close()
        
    def moveRel(self,_x,_y):
        
        self.appendSection(svg_path.move_relative,x=_x,y=_y)
        
    def moveAbs(self,_x,_y):
        
        self.appendSection(svg_path.move_absolute,x=_x,y=_y)
        
    def lineRel(self,_x,_y):
        
        self.appendSection(svg_path.line_relative,x=_x,y=_y)
        
    def lineAbs(self,_x,_y):
        
        self.appendSection(svg_path.line_absolute,x=_x,y=_y)
        
    def horizontalRel(self,_x):
        
        self.appendSection(svg_path.horiz_relative, x=_x)
        
    def verticalRel(self,_y):
        
        self.appendSection(svg_path.verti_relative, y=_y)
        
    def circularArcRel(self,_x,_y,_r,long_arc = False, clockwise = False):
        
        self.appendSection(svg_path.circ_arc_relative, x=_x,y=_y,r=_r,large_arc=int(long_arc),sweep=int(clockwise))   
        
        
    def close(self):
        
        if not self.closed():
            self.path_text.append("z")
        
    def closed(self):
        
        if len(self.path_text) == 0:
            
            return False
        
        return self.path_text[-1] == "z"
    
 
    def unclose(self):
        
        if self.closed():
            
            self.path_text.pop()
            return True
        
        return False
    
    def get_path_string(self):
        
        return ''.join(self.path_text)
    
    def move_get_path_string(self,x,y):
        
        return svg_path.move_absolute.format(x,y).join(self.path_text)
            
            

         
# Use-case example

if __name__ == "__main__":
            
    lens1 = thinLens('lens1')
    lens2 = thinLens('lens2')
    lens3 = thinLens('lens3')
    
  #  dwg = svgwrite.Drawing('lens-eg.svg',size=(640,480),profile='tiny')
   # inkscape = Inkscape(dwg)
    
    lens1.diameter = 40
    lens1.f = 60
    lens1.makePlano()
    
    lens2.r0 = 50
    lens2.r1 = -100
    lens2.diameter = 25
    lens2.fFromLensmaker()
    
    lens3.r0 = 300
    lens3.r1 = -80
    lens3.diameter = 30
    lens3.constrainDiameter()
    lens3.fFromLensmaker()
    
    print("Lens 1")
    print(lens1.f)
    print(lens1.diameter)
    print("Lens 2")
    print(lens2.f)
    print(lens2.diameter)
    print("lens 3")
    print(lens3.f)
    print(lens3.diameter)
    
    
    optics = opticalSystem()
    
    lens1.x  =  10
    lens2.x  =  40
    lens3.x  = 110
    
    optics.addElement(lens1) 
    optics.addElement(lens2)
    optics.addElement(lens3)
    
    #opticLayer = inkscape.layer(label="optics")
    #dwg.add(opticLayer)
    

   # optics.draw(dwg,layer=opticLayer,axisLimits=(-300,120))        
    
    project = tracingProject(optics)
    
    project.setObject(-120)
    project.setInputPlane(-200,relativeTo='firstElement')

    #optics.draw(dwg,layer=opticLayer,axisLimits=(-200,120))        

    project.setOutputPlane(30,relativeTo='lastElement')
    
    project.solveAll()
    
    angles = np.arange(-0.2,0.1,0.02)

    project.solveAll()
    
    project.addTracesAngleRange(10,angles,group=1)
    project.addTracesAngleRange(-10,-1*angles,group=2)
    project.traceAll()
    #project.drawTraces(dwg,inkscape=inkscape)
    #project.drawObject(dwg,inkscape=inkscape)
    
    project.report()
    
    #dwg.save()
    
    
    drawing = lensrender()
    
    drawing.drawOpticalSystem(optics)
    
    for ray in project.traces:
        
        drawing.drawRayTrace(ray)
        
    drawing.drawOpticalAxis()
    
    drawing.drawOpticElement(project.entrancePupil, "pupils")
    drawing.drawOpticElement(project.exitPupil, "pupils")
    drawing.drawOpticElement(project.entranceWindow, "windows")
    drawing.drawOpticElement(project.exitWindow, "windows")
    
    drawing.save_drawing()
    
    
    
    """
    trace1 = rayTrace(-10,10,0)
    
    trace1.propagateThrough(optics)
    
    trace1.draw(dwg)
    
    print(dwg.tostring())"""
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    