# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 15:36:04 2020

@author: James Hoyland

lensdraw.py

lensdraw is a library for generating to-scale SVG ray diagrams of simple optical systems.

Features:
    
    Create linear arrays of lenses, appertures and other components
    Accurately trace light rays through the lens optics using optical matrices
    Perform matrix calculations on systems of lenses
    Locate stops and pupils
    Output diagrams of optical systems and rays in SVG format

"""

# Here's  acomment
# Messing with some stuff

import svgwrite
from svgwrite.extensions import Inkscape
import numpy as np
import math

DEFAULT_INDEX = 2.3

# The Lensmaker's Forumula for calculating thin lens focal length from curvature and index
def lensmaker(r0,r1,n,n0=1.0):
    lensPower = (n-n0) * (1/r0 - 1/r1) / n0
    return 1/lensPower
g
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


# Class to hold entire optical optics.
# Elements are added to a list which is then sorted by x position of the element

class opticalSystem:
    
    DEFAULT_DISPLAY_HEIGHT = 140
    DEFAULT_DISPLAY_WIDTH = 960
    DEFAULT_SCALE_POSITIONING = 3  # 1 pixel = 1 mm, used for positioning of planes
    DEFAULT_SCALE_ELEMENTS = 3  # 1 pixel = 1 mm, used for visual sizing of lenses and other elements
    DEFAULT_SCALE_CURVATURE = 3

    def __init__(self):
        self.display_height = opticalSystem.DEFAULT_DISPLAY_HEIGHT
        self.display_width = opticalSystem.DEFAULT_DISPLAY_WIDTH
        self.scale_position = opticalSystem.DEFAULT_SCALE_POSITIONING 
        self.scale_elements = opticalSystem.DEFAULT_SCALE_ELEMENTS 
        self.scale_curvature = opticalSystem.DEFAULT_SCALE_CURVATURE 
        self.axis_height = 0.5 * self.display_height
        self.elements = []
        #self.objectPlane = opticalPlane("__object__")
        #self.imagePlane = opticalPlane("__image__")
        #self.objectPlane.x = -np.inf
        
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
            
    def addApperture(self,name,x,d,afterElement=None):
        
        app = apperture(name)
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
        
    
    
        
    def removeElement(self,name):
        el = self.getElementByName(name)
        
        if el is not None:
            self.elements.remove(el)
        
    def draw(self,svgDrawing, layer=None, opticalAxis=True, axisLimits=None):
           
        if opticalAxis:
            
            if axisLimits is None:
                axisLimits = (0,self.display_width)
            ax = svgDrawing.line(start=(axisLimits[0]*self.scale_position,self.axis_height), end=(axisLimits[1]*self.scale_position,self.axis_height), stroke="#a0a0a0", stroke_dasharray='2,2')
            if layer is None:
                svgDrawing.add(ax)
            else:
                layer.add(ax)
                
        
        for element in self.elements:
            if not element.excludeFromDrawing:
                element.draw(svgDrawing,layer)
                
                
    def setDrawingScale(self,x,y):
        self.scale_position = x
        self.scale_elements = y
            
                
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

        
        if toElement == '__all__':
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
    
    def propagateRay(self,ray,endOnBlocked = True,startPlane = "__all__", endPlane = "__all__"):
        
        raySegments = [ray]
        
        started = startPlane == "__all__"
        
        i=0
        
        for plane in self.elements:
            L = plane.x - raySegments[-1].x
            if L > 0:
                newVector = translationMatrix(L) * raySegments[-1].vector
                
                newVector = plane.getMatrix() * newVector
                newElement = rayElement(plane.x,newVector)
                
                    
                newElement.blocked = raySegments[-1].blocked or plane.rayBlocked(abs(newVector[0,0]))
                    
                raySegments.append(newElement)
                
                if not started:
                    i = i+1
                    if startPlane == plane.name:
                        started = True
                
                if endOnBlocked and newElement.blocked:
                    break
                
                if plane.name == endPlane:
                    break
                         
        return raySegments[i:]
    

    

class opticalPlane:
    
    
    def __init__(self,name,label=None):
        if label is None:
            self.label = name
        else:
            self.label = label
            
                
                
        self.name = name
        self.x = 0
        self.optics = None
        
        self.stroke = "#090909"
        self.stroke_width = 0.5
        self.fill = "none"
        self.stroke_dasharray = "8,8"
        
        self.excludeFromDrawing = False
        
        self.physical = False
        
        self.diameter = np.inf
        
    def __str__(self):
        
        ret = "Optical Plane: {:} @{:} diameter={:}".format(self.name,self.x,self.diameter)
        return ret
        
    # Transfer matrix for ray calculations. Default is the identity matrix
        
    def getMatrix(self):
        return identityMatrix()
    
    def processRay(self,ray):
        return self.getMatrix().dot(ray)
    
    def rayBlocked(self,rayHeight): 
        return False
    
    def draw(self,svgDrawing,layer=None):
        
        pathString = self.getSVGPathString()
        locString = self.getSVGLocationString()
        
        drawString = locString + pathString
        
        path = svgDrawing.path(d=drawString, stroke_linejoin = 'round', stroke_linecap = 'round', stroke = self.stroke, stroke_width = self.stroke_width, fill = self.fill, stroke_dasharray = self.stroke_dasharray)
        if layer is None:
            svgDrawing.add(path)
        else:
            layer.add(path)
        
    def getSVGLocationString(self):
        
        xpos = self.x * self.optics.scale_position
        ypos = self.optics.axis_height
        
        return "M {:0.2f},{:0.2f} ".format(xpos,ypos)
    
    def getSVGPathString(self):
        if self.optics is None:
            h = opticalSystem.DEFAULT_DISPLAY_HEIGHT
        else:
            h = self.optics.display_height
            
        return "m 0,{start:} l 0,{distance:} ".format(start=-0.5*h,distance=h)

class physicalObject(opticalPlane):
    
    def __init__(self,name,label=None):
        
        super().__init__(name,label)
        
        self.stroke = "black"
        self.stroke_width = 1
        self.fill = "black"
        self.stroke_dasharray = "100,0"
        
        self.physical = True
        
        self.diameter = 25.4
        
        self.customDrawFunction = None
        
        self.drawPlane = False
        
    
    def rayBlocked(self,rayHeight):
        return rayHeight < (0.5 *  self.diameter)
        
    def draw(self,svgDrawing,layer=None):
        
        if self.drawPlane:
            
            pathString = super().getSVGPathString()
            locString = self.getSVGLocationString()
        
            drawString = locString + pathString
        
            path = svgDrawing.path(d=drawString, stroke = self.stroke, stroke_width = self.stroke_width, fill = self.fill, stroke_dasharray = self.stroke_dasharray)
            
            if layer is None:
                svgDrawing.add(path)
            else:
                layer.add(path)
        
        if self.customDrawFunction is not None:
            
            self.customDrawFunction(svgDrawing,layer)
            
        else:
            
            super().draw(svgDrawing,layer)
            
            
    def getSVGPathString(self):
        
        h = -self.diameter
        
        h = h*self.optics.scale_elements
        
        sgn = h/abs(h)
        
        arrowPoints = [a * sgn * self.optics.scale_elements for a in (-4,-6,8,0,-4,6)]

        
        shaft = "m 0,{start:} l 0,{distance:} ".format(start=-0.5*h,distance=h)   
        head = "l {},{} l {},{} l {},{} ".format(*arrowPoints)

        return shaft + head
           
class  imagePlane(physicalObject) :

    def __init__(self,name,label=None):
        super().__init__(name,label)
        self.physical = False   
        self.mag = 1
        
    
    def rayBlocked(self,rayHeight): 
        return False

class beamBlock(opticalPlane):
    
    def __init__(self,name,label=None):
        super().__init__(name,label)
        self.stroke_width = 6
        self.stroke_dasharray = "100,0"
        self.physical = True
        
    def rayBlocked(self,rayHeight):
        return True
    
            
class apperture(opticalPlane):
    
    def __init__(self,name,label=None):
        super().__init__(name,label)
        self.diameter = 25.4
        self.display_thickness = 5
        self.stroke  = "#000"
        self.stroke_width = 1.5
        self.stroke_dasharray = "100,0"
        self.physical = True
        
        
    def __str__(self):
        
        ret = "Physical aperture: {:} @{:} diameter={:}".format(self.name,self.x,self.diameter)
        return ret
        
    def rayBlocked(self,rayHeight):
        return rayHeight > (0.5 *  self.diameter)
    
    def getSVGPathString(self):
        if self.optics is None:
            h = opticalSystem.DEFAULT_DISPLAY_HEIGHT
        else:
            h = self.optics.display_height
            disp_diameter = self.diameter * self.optics.scale_elements
            
        if disp_diameter > h-5:
            h = disp_diameter + 5
            
        partString = "m 0,{start:} l 0,{distance} m {t:},0 l {d:},0 "
        
        dist = (0.5* (h - disp_diameter),h-2*self.stroke_width)[h<disp_diameter]
        
        topString = partString.format(start = -0.5 * h, distance = dist, t = -0.5 * self.display_thickness, d = self.display_thickness)
        moveString = "m {},{} ".format(-0.5 * self.display_thickness, 0.5 * disp_diameter)
        bottomString = partString.format(start = 0.5 * h, distance = -dist, t = -0.5 * self.display_thickness, d = self.display_thickness)

        return topString + moveString + bottomString
    
class imageSensor(apperture):
    
    def __init__(self,name,label=None):
        super().__init__(name,label)
        self.stroke = "#000000"
        self.stroke_width = 1
        self.fill = '#808080' 
        
                
    def __str__(self):
        
        ret = "Image sensor: {:} @{:} diameter={:}".format(self.name,self.x,self.diameter)
        return ret
        
    def rayBlocked(self,rayHeight):
        return rayHeight < (0.5 * self.diameter)
    
    def getSVGPathString(self):
        
        disp_diameter = self.diameter * self.optics.scale_elements
        thick = 8
        
        stng = "m 0,{st:} l {t:},0 l 0,-{d:} l -{t:},0 z".format(st = disp_diameter * 0.5, t = thick, d = disp_diameter)
        
        return stng
    
class appertureImage(apperture):
        
    def __init__(self,name,label=None):
        super().__init__(name,label)
        self.stroke  = "#cc0000"
        self.physical = False
        
            
    def __str__(self):
        
        ret = "Effective apperture: {:} @{:} diameter={:}".format(self.name,self.x,self.diameter)
        return ret

class thinLens(apperture):
    
    def __init__(self,name,label=None):
        super().__init__(name,label)
        self.n = DEFAULT_INDEX
        self.f = 100
        self.r0 = bilenscurvature(self.f,self.n)
        self.r1 = -self.r0        
        self.stroke = "#000000"
        self.stroke_width = 0.5
        self.fill = "#CCCCCC"
        self.display_thickness = 2
                
    def __str__(self):
        
        ret = "Thin lense: {:} @{:} diameter={:} f={:}".format(self.name,self.x,self.diameter,self.f)
        return ret
    
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
    
    def constrainDiameter(self):
        
        largestDiameter = 2 * min(abs(self.r0),abs(self.r1))
        
        self.diameter = min(largestDiameter,self.diameter)
        
    
    def setf(self,f,form='plano',flatLeft=False):
        
        self.f = f
        
        if form == 'plano':
            radii = (planolenscurvature(self.f, self.n),np.inf)
            self.r0 = radii[flatLeft]
            self.r1 = radii[not flatLeft]
        elif form == 'bi':
            self.r0 = bilenscurvature(self.f,self.n)
            self.r1 = -self.r0
        else:
            self.r0 = bilenscurvature(self.f,self.n)
            self.r1 = -self.r0
            
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
        self.r0 = r
        self.r1 = -r
            
        self.constrainDiameter()
            
    
    def getSVGPathString(self):
        curvedSurfaceString = 'a{rx:0.3f},{rx:0.3f} 0 0 {sweep:} 0,{diameter:0.3f} '
        lensEdgeString = 'l{thickness:0.3f},0 '
        flatSurfaceString = 'l0,{diameter:0.3f} '
        
        DLens = self.diameter * self.optics.scale_elements
        
        if abs(self.r0) == np.inf:
            lsag = 0
            leftSurface = flatSurfaceString.format(diameter=DLens)
        else:
            R0 = self.r0 * self.optics.scale_curvature
            if abs(R0) < 0.5 * DLens:
                R0 = math.copysign(0.5*DLens,R0)
            lsag = math.copysign(sagitta(abs(R0),DLens),R0)
            leftSurface = curvedSurfaceString.format(rx=abs(R0),sweep=(0,1)[R0<0],diameter=DLens)
            
        if abs(self.r1) == np.inf:
            rsag = 0
            rightSurface = flatSurfaceString.format(diameter=-DLens)
        else:            
            R1 = self.r1 * self.optics.scale_curvature
            if abs(R1) < 0.5 * DLens:
                R1 = math.copysign(0.5*DLens,R1)
            rsag = math.copysign(sagitta(abs(R1),DLens),-R1)
            rightSurface = curvedSurfaceString.format(rx=abs(R1),sweep=(1,0)[R1<0],diameter=-DLens)
            
        rimThickness = self.display_thickness
        centerThickness = lsag + rsag + self.display_thickness
        
        if centerThickness < self.display_thickness:
            rimThickness += (rimThickness - centerThickness)
            
        
        topEdgeString = lensEdgeString.format(thickness = rimThickness)
        startPointString = 'm{x0:.3f},{y0:.3f} '.format(x0=-0.5*rimThickness,y0=-0.5*DLens)
        
        return startPointString + leftSurface + topEdgeString + rightSurface + 'z' 
         

class rayElement:

    def __init__(self,x,vector: np.array):
        self.x = x
        self.vector = vector
        self.blocked = False
        self.excluded = False
        
    def height(self):
        return self.vector[0,0]
    
    def angle(self):
        return self.vector[1,0]
    
    def xy(self,scale=(1,1)):
        return (float(self.x * scale[0]),float(self.vector[0,0] * scale[1]))
        

class rayTrace:
    
    def __init__(self,x,h,angle,group=0):
        
        self.group = group
        self.h = h 
        self.angle = angle
        self.x = x
        self.trace = []
        self.stroke_dasharray = "100,0"
        self.stroke_dasharray_virtual = "1,1"
        self.stroke = "#FF0000"
        self.stroke_blocked = "#0000FF"
        self.stroke_width = 0.5
        self.optics = None
        
        
        
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
        
        startRay = rayElement(x0,np.array([[self.h+dh],[self.angle]]))
        self.trace.clear()
        self.trace = optics.propagateRay(startRay,toBlocked,startPlane,endPlane)
        
        
        
        if clipEnd is not None:
            if not(toBlocked and self.trace[-1].blocked):
                L = clipEnd - self.trace[-1].x
                if L > 0:
                    newVector = translationMatrix(L) * self.trace[-1].vector
                    newElement = rayElement(clipEnd,newVector)
                    newElement.blocked = self.trace[-1].blocked
                    self.trace.append(newElement)
                
        self.optics = optics
                
                
        
    def draw(self,svgDrawing,layer = None):
        
        if self.trace is not None:
            
           
            previous = self.trace[0]
            scaleTup = (self.optics.scale_position,self.optics.scale_elements)
                        
            for element in self.trace[1:]:
                                
                x1, y1 = previous.xy(scale = scaleTup)
                x2, y2 = element.xy(scale = scaleTup)
                                
                y1 = self.optics.axis_height - y1
                y2 = self.optics.axis_height - y2
                
                strk =  (self.stroke,self.stroke_blocked)[int(previous.blocked)]
                
                line = svgDrawing.line( start = (x1,y1),end = (x2,y2),stroke =strk, stroke_width = self.stroke_width,stroke_dasharray = self.stroke_dasharray)
                
                if layer is None:
                    svgDrawing.add(line)
                else:
                    layer.add(line)
            
                previous = element
        
    def drawVirtualRay(self,svgDrawing,layer=None,toPlane=0):
        
        if self.trace is not None:
            
            scaleTup = (self.optics.scale_position,self.optics.scale_elements)
            x1, y1 = self.trace[-2].xy(scale = scaleTup)
            x2 = toPlane * self.optics.scale_position
            y2 = (self.trace[-2].height() + (toPlane - self.trace[-2].x) * self.trace[-2].angle()) * self.optics.scale_elements
            
            
            y1 = self.optics.axis_height - y1
            y2 = self.optics.axis_height - y2
            
            line = svgDrawing.line( start = (x1,y1),end = (x2,y2),stroke =self.stroke, stroke_width = self.stroke_width,stroke_dasharray = self.stroke_dasharray_virtual)
                
            if layer is None:
                svgDrawing.add(line)
            else:
                layer.add(line)
        
'''
tracingProject:
    collects together the optics along with a specific object and associated tracing and calculation tasks.
    it will use the optics matrix to calculate image distances,
'''



class tracingProject:
    
    DEFAULT_OUTPUT_LOCATION = 100
    DEFAULT_INPUT_LOCATION = -25.4
    DEFAULT_TRACE_COLOR = 'red'
    
    ENTRANCE_WINDOW_COLOR = '#E6F0B4'
    EXIT_WINDOW_COLOR = '#C8F0B4'
    ENTRANCE_PUPIL_COLOR = '#F0DCB4'
    EXIT_PUPIL_COLOR = '#F0BEB4'
    AS_COLOR = '#E00000'
    FS_COLOR = '#00E000'
    
    def __init__(self,optics):
        
        self.optics = optics
        self.object = None
        self.image = None
        self.intermediateImages = []
        self.traces = []
        self.entrancePupil = None
        self.exitPupil = None
        self.appertureStop = None
        self.fieldStop = None
        self.sysMatrixToAS = None
        self.exitWindow = None
        self.entranceWindow = None
        
        self.valid = False
        
        self.angularApperture = 0
        self.FOV = math.pi
        
        self.inputLocation = 0
        self.outputLocation = 0
        
        self.traceGroup = 0
        
    # SETUP
    
    def setObject(self, x, size = 25.4):
        
        self.valid = False
                
        self.object = physicalObject('object')
        
        self.object.x = -x
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
    To find the apperture stop. Calculate the optics matrix up to each successive plane. For objects not at infinity
    include the translation matrix from the object plane to the first element.  For each plane calculate the input angle (ai)
    from the aperture diameter (d) at the plane and the partial matrix as
    
    ai = d / (2*B)
    
    The element producing the smallest angle is the aperture stop.
    If element B of the partial matrix is zero then the plane is conjugate with the object plane and cannot be the 
    apperture stop (though it can be the field stop)
    
    If the object is at infinity the same procedure is done starting with the first physical element of the optics. This time
    the input ray height parallel to the axis (hi) is calculated is
    
    hi = d / (2*A)
    
    Again if element A is zero there is an intermediate image at the plane and so it is not the aperture stop.
    """
    
    
    
    def findAppertureStop(self):
        
        self.angularApperture = np.inf
        
        finiteObject = abs(self.object.x) != np.inf
        
        initialTransferMatrix = (identityMatrix(),translationMatrix(abs(self.object.x)))[finiteObject]
        
        for element in self.optics.elements:
            
            if element.diameter < np.inf:
                
                syMx = self.optics.getSystemMatrix(toElement = element.name) * initialTransferMatrix
                mxel = syMx[0,(0,1)[finiteObject]] # Select element A or B
                
          #      print(syMx)
         #       print(mxel)
                
                if mxel != 0:
                
                    r0 = abs(element.diameter / (2*mxel))
                
                    if r0 < self.angularApperture:
                        self.appertureStop = element
                        self.angularApperture = r0

            
    def findEntrancePupil(self):
        
        if self.appertureStop is not None:
            
            self.entrancePupil = appertureImage('__entrance_pupil__')
            self.entrancePupil.optics = self.optics
            self.entrancePupil.stroke = tracingProject.ENTRANCE_PUPIL_COLOR
            firstElement = self.optics.elements[0]
            
            if self.appertureStop.name == firstElement.name:
                          
                self.entrancePupil.x = firstElement.x
                self.entrancePupil.diameter =  firstElement.diameter
                
            else:
            
                syMx = self.optics.getSystemMatrix(fromElement = self.appertureStop.name ,backward = True,inclusive = (False,True))
                
                si,mag = getImageDistance(syMx)
                
      #          print("Entrance pupil image distance {:}".format(si))
                            
                self.entrancePupil.x = firstElement.x - si
                self.entrancePupil.diameter =  abs(self.appertureStop.diameter * (syMx[0,0] + si * syMx[1,0]))
            
            return True
        
        return False
    
    
    def findExitPupil(self):
                        
        if self.appertureStop  is not None:
            
            self.exitPupil = appertureImage('__exit_pupil__')
            self.exitPupil.optics = self.optics
            self.exitPupil.stroke = tracingProject.EXIT_PUPIL_COLOR
            lastElement = self.optics.elements[-1]
            
            if self.appertureStop.name == lastElement.name:
                
                self.exitPupil.x = lastElement.x          
                self.exitPupil.diameter = lastElement.diameter
            
            else:
            
                syMx = self.optics.getSystemMatrix(fromElement = self.appertureStop.name,inclusive = (False,True))
                
                si,mag = getImageDistance(syMx)
                
                self.exitPupil.x = lastElement.x + si            
                self.exitPupil.diameter = abs(self.appertureStop.diameter * (syMx[0,0] + si * syMx[1,0]))
            
            return True
        
        return False
    
                       
    def findFieldStop(self):
        
        aF = np.inf
        
       # finiteObject = abs(self.object.x) != np.inf
        
        #initialTransferMatrix = (identityMatrix(),translationMatrix(abs(self.object.x)))[finiteObject]
        
    #    print("Finding field stop")
        
        for element in self.optics.elements:
            
     #       print("Checking element: " + element.name)
            
            if element.diameter < np.inf:
                
                if not(element.name == self.appertureStop.name):
                
                    direction = element.x < self.appertureStop.x
                    
   #                 if direction:
    #                    print("Before AS")
     #               else:
      #                  print("After AS")
                    
                    syMx = self.optics.getSystemMatrix(fromElement = self.appertureStop.name, toElement = element.name,backward = direction, inclusive = (False,False))
                       
   #                 print("System matrix: {:}".format(syMx))
                    
                    mxel = syMx[0,1] 
                    
   #                 print("Element B = {:}".format(mxel))
                
                    if mxel != 0:
                                            
                        r0 = abs(element.diameter / (2*mxel)) % (2*math.pi)
                        
                        print("Diameter = {:}   Angle = {:}".format(element.diameter,r0))
                    
                        if r0 < aF:
                            
     #                       print("New Field Stop")
                            self.fieldStop = element
                            aF = r0
                            
     #           print("Skipping: element is AS")
                            
     #       else:
                
    #            print("Skipping: infinite diameter")
            
            
                
    def findEntranceWindow(self):
        
        if self.fieldStop is not None:
            
            self.entranceWindow = appertureImage('__entrance_window__')
            self.entranceWindow.optics = self.optics
            self.entranceWindow.stroke = tracingProject.ENTRANCE_WINDOW_COLOR
            firstElement = self.optics.elements[0]
            
            if self.fieldStop.name == firstElement.name:
                          
                self.entranceWindow.x = firstElement.x
                self.entranceWindow.diameter =  firstElement.diameter
                
            else:
            
                syMx = self.optics.getSystemMatrix(fromElement = self.fieldStop.name ,backward = True,inclusive = (False,True))
                
                si,mag = getImageDistance(syMx)
                
        #        print("Entrance window image distance {:}".format(si))
                            
                self.entranceWindow.x = firstElement.x - si
                
                if abs(si) == np.inf:
                    # angular diameter for infinite entrance window distance
                    self.entranceWindow.diameter =  abs(self.fieldStop.diameter * syMx[1,0])  
                else:
                    self.entranceWindow.diameter =  abs(self.fieldStop.diameter * (syMx[0,0] + si * syMx[1,0]))
            
            return True
        
        return False   
    
                    
    def findExitWindow(self):
        
        if self.fieldStop is not None:
            
            self.exitWindow = appertureImage('__exit_window__')
            self.exitWindow.optics = self.optics
            self.exitWindow.stroke = tracingProject.EXIT_WINDOW_COLOR
            lastElement = self.optics.elements[-1]
            
            if self.fieldStop.name == lastElement.name:
                          
                self.exitWindow.x = lastElement.x
                self.exitWindow.diameter =  lastElement.diameter
                
            else:
            
                syMx = self.optics.getSystemMatrix(fromElement = self.fieldStop.name,inclusive = (False,True))
                
                si,mag = getImageDistance(syMx)
                
        #        print("Entrance window image distance {:}".format(si))
                            
                self.exitWindow.x = si + lastElement.x
                
                if abs(si) == np.inf:
                    # angular diameter for infinite exit window distance
                    self.exitWindow.diameter =  abs(self.fieldStop.diameter * syMx[1,0])  
                else:
                    self.exitWindow.diameter =  abs(self.fieldStop.diameter * (syMx[0,0] + si * syMx[1,0]))
            
            return True
        
        return False   



    def getFOV(self):
        
        dEW = 0.5* self.entranceWindow.diameter
        dEP = 0.5* self.entrancePupil.diameter

        xEW = self.entranceWindow.x
        xEP = self.entrancePupil.x
        
        if abs(xEW) == np.inf:
            
            ao = dEW
            
        else:
        
            dx = abs(xEW - xEP)
            
       #     print("xEW = {:}  xEP = {:}".format(xEW,xEP))
       #     print("dEW = {:}  dEP = {:}".format(dEW,dEP))
            
            #ao = math.atan(dEW / dx)
            
            ao = dEW / dx
            
      #      print("dx = {:}   ao = {:}".format(dx,ao))
        
        self.FOV = ao * 2

    def solveAll(self):
        
        self.calculateImage()
        self.findAppertureStop()
        self.findExitPupil()
        self.findEntrancePupil()
        self.findFieldStop()
        self.findEntranceWindow()
        self.findExitWindow()
        self.getFOV()
        


# TRACING
 
    def addTrace(self,h,a,group=0,color=DEFAULT_TRACE_COLOR):
        
        if h == 'top':
            h = self.object.diameter * 0.5
        if h == 'bottom':
            h = -self.object.diameter * 0.5
            
        self.traces.append(rayTrace(self.object.x,h,a,group))
        self.traces[-1].stroke = color
        
    def addTracesAngleRange(self,h,angles,group=0,color=DEFAULT_TRACE_COLOR):
        
        for angle in angles:                      
            self.addTrace(h,angle,group,color)
                
    def addTracesHeightRange(self,hs,a,group=0,color=DEFAULT_TRACE_COLOR):
        
        for h in hs:                      
            self.addTrace(h,a,group,color)
            
    def addTracesFillFirstElement(self,h,fill_factor=0.9,numberStep=10,method='number',group=0,color=DEFAULT_TRACE_COLOR):
        
                
        if h == 'top':
            h = self.object.diameter * 0.5
        if h == 'bottom':
            h = -self.object.diameter * 0.5
        
        d = fill_factor * self.optics.elements[0].diameter * 0.5
        dx = - self.object.x
        
        amin = - (d+h) / dx
        amax = (d-h) / dx
        
    #    print(amax)
    #    print(amin)
        
        if method == 'step':
            
            numberStep =math.ceil((amax - amin) / numberStep)
 
        
        angles = np.linspace(amin,amax,numberStep )
        
        self.addTracesAngleRange(h, angles, group, color)
        
        
    def addMarginalRay(self,group=0,negative=False,color=DEFAULT_TRACE_COLOR):
        
                
        if self.appertureStop is not None:
            
            aa = 0.99*self.angularApperture
            
            if negative:
                
                aa = -aa
            
            if abs(self.object.x) == np.inf:
                
                self.addTrace(aa,0,group,color)
                
            else:
                
                self.addTrace(0,aa,group,color)      
        
    def addFullFieldChiefRay(self,group=0,negative=True,rays=1,color=DEFAULT_TRACE_COLOR):
        
        if self.entrancePupil is not None:
            
            aa = self.FOV * 0.5 * 0.99
                
            # Object is at infinity and entrancePupil is behind the first element
            # Trace the chief ray back to the first element to give it a height
            # This is needed becuase the ray tracer measures initial ray height for 
            #infinite objectson the front element and calculates back. May need to 
            # rethink this
                
            if self.object.x == -np.inf and self.entrancePupil.x > self.optics.elements[0].x:
                
                h = ( self.entrancePupil.x - self.optics.elements[0].x ) * aa
                
            else:
                    
                h = abs(self.entrancePupil.x - self.object.x) * aa
                
     ##       print("Chief {:}  {:}".format(aa,h))
                
            if negative:
                
                aa = -aa
                
            else:
                
                h = -h
                
            if rays>1:
                
                ang = np.linspace(aa + self.angularApperture, aa - self.angularApperture, rays)
                
            else:
                
                ang = [aa]
            
         #   print("Chief {:}  {:}".format(aa,h))
                
           ## dwp = abs(trace.entrancePupil.x - trace.entranceWindow.x)

          ##  a2 = 0.5 * trace.entranceWindow.diameter / dwp

##            h2 = - dx * a2

            for a in ang:
                
                self.addTrace(h,a,group,color)
    
    def traceAll(self,toBlocked = True):
        
        for trace in self.traces:
            
            trace.propagateThrough(self.optics,clipStart = self.inputLocation, clipEnd = self.outputLocation, toBlocked = toBlocked)
       

# DRAWING 

    def report(self):
        
        print("Optical system:")
        
        for e in self.optics.elements:
            
            print(e)
            
        print("Object @{:}".format(self.object.x))
        print("Final Image @{:}".format(self.image.x))
        
        print("Aperture stop = {:}".format(self.appertureStop.name))
        print("Field stop = {:}".format(self.fieldStop.name))
        
        degFOV = math.degrees(self.FOV)
        
        print("Field of View: {:}".format(degFOV))
        
        print(self.entrancePupil)
        print(self.exitPupil)
        print(self.entranceWindow)
        print(self.exitWindow)
              
              
            
        

    def drawObject(self,svgDrawing,layer = None, inkscape = None):
        
        if self.object is not None:
            
            if self.object.x > self.inputLocation:
                
                if inkscape is not None:
                    layer = inkscape.layer('object')
                    svgDrawing.add(layer)
                    
                self.object.draw(svgDrawing,layer)
                   
    def drawImage(self,svgDrawing,layer = None, inkscape = None):
        
        if self.object is not None:
            
            if self.image.x <= self.outputLocation:
                
                if inkscape is not None:
                    layer = inkscape.layer('image')
                    svgDrawing.add(layer)
                
                self.image.draw(svgDrawing,layer)
                
    def drawTraces(self,dwg,layer=None,inkscape=None,drawVirtuals=False):
        
        g=-1
        vlayer = layer
        self.traces.sort(key = lambda t: t.group)
        
        if abs(self.image.x) == np.inf:
            drawVirtuals = False
        
        for trace in self.traces:
            if inkscape is not None:
                if trace.group > g:
                    g = trace.group    
                    layerLabel = "rayGroup{}".format(g)    
                    layer = inkscape.layer(layerLabel)
                    dwg.add(layer)
                    if drawVirtuals:
                        vlayer = inkscape.layer(layerLabel+"Virtual")
                        dwg.add(vlayer)
                        
            
            
            trace.draw(dwg,layer)
            
            if drawVirtuals:
                trace.drawVirtualRay(dwg,vlayer,toPlane=self.image.x)
                
    def drawApertureStop(self,svgDrawing,layer = None, inkscape = None):
           
        if self.appertureStop is not None:
            
            asDraw = apperture("AS")
            asDraw.x = self.appertureStop.x
            asDraw.optics = self.optics
            asDraw.diameter = self.appertureStop.diameter
            asDraw.stroke = tracingProject.AS_COLOR
                
            if inkscape is not None:
                layer = inkscape.layer('AS')
                svgDrawing.add(layer)
            
            asDraw.draw(svgDrawing,layer)    
            
                            
    def drawFieldStop(self,svgDrawing,layer = None, inkscape = None):
           
        if self.fieldStop is not None:
            
            fsDraw = apperture("FS")
            fsDraw.x = self.fieldStop.x
            fsDraw.optics = self.optics
            fsDraw.diameter = self.fieldStop.diameter
            fsDraw.stroke = tracingProject.FS_COLOR
                
            if inkscape is not None:
                layer = inkscape.layer('FS')
                svgDrawing.add(layer)
            
            fsDraw.draw(svgDrawing,layer)    
            
            
    def drawExitPupil(self,svgDrawing,layer = None, inkscape = None):
           
        if self.object is not None:
            
            if self.exitPupil.x <= self.outputLocation:
                
                if inkscape is not None:
                    layer = inkscape.layer('exitPupil')
                    svgDrawing.add(layer)
                
                self.exitPupil.draw(svgDrawing,layer)     
        
              
            
    def drawEntrancePupil(self,svgDrawing,layer = None, inkscape = None):
           
        if self.object is not None:
            
            if self.entrancePupil.x > self.inputLocation:
                
                if inkscape is not None:
                    layer = inkscape.layer('entrancePupil')
                    svgDrawing.add(layer)
                
                self.entrancePupil.draw(svgDrawing,layer)                
            
    def drawExitWindow(self,svgDrawing,layer = None, inkscape = None):
           
        if self.object is not None:
            
            if self.exitWindow.x <= self.outputLocation:
                
                if inkscape is not None:
                    layer = inkscape.layer('exitWindow')
                    svgDrawing.add(layer)
                
                self.exitWindow.draw(svgDrawing,layer)                
            
    def drawEntranceWindow(self,svgDrawing,layer = None, inkscape = None):
           
        if self.object is not None:
            
            if self.entranceWindow.x > self.inputLocation:
                
                if inkscape is not None:
                    layer = inkscape.layer('entranceWindow')
                    svgDrawing.add(layer)
                
                self.entranceWindow.draw(svgDrawing,layer)              
                

    

                
            

    
 
         
# Use-case example

if __name__ == "__main__":
            
    lens1 = thinLens('lens1')
    lens2 = thinLens('lens2')
    lens3 = thinLens('lens3')
    
    dwg = svgwrite.Drawing('lens-eg.svg',size=(640,480),profile='tiny')
    inkscape = Inkscape(dwg)
    
    lens1.diameter = 40
    lens1.f = 60
    lens1.makePlano()
    
    lens2.r0 = 30
    lens2.r1 = -35
    lens2.diameter = 10
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
    
    opticLayer = inkscape.layer(label="optics")
    dwg.add(opticLayer)
    
    optics.draw(dwg,layer=opticLayer,axisLimits=(-200,120))        
    
    project = tracingProject(optics)
    
    project.setObject(-50)
    project.setInputPlane(-100,relativeTo='firstElement')
    project.setOutputPlane(30,relativeTo='lastElement')
    
    angles = np.arange(-0.2,0.1,0.02)
    
    project.addTracesAngleRange(20,angles,group=1,color='green')
    project.addTracesAngleRange(-20,-1*angles,group=2,color='red')
    project.traceAll()
    project.drawTraces(dwg,inkscape=inkscape)
    
    dwg.save()
    
    
 
"""    trace1 = rayTrace(-10,10,0)
    
    trace1.propagateThrough(optics)
    
    trace1.draw(dwg)
    
    print(dwg.tostring())"""
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    