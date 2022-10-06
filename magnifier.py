import lensdraw
import numpy as np

camlens = lensdraw.thinLens('lens')

camlens.f = 27
camlens.diameter = 10
camlens.makeBi()
camlens.x = 0

ccd = lensdraw.aperture('camera')

ccd.diameter = 36.74
ccd.x = 28


eyepiece = lensdraw.thinLens('eyepiece')
eyepiece.diameter = 50
eyepiece.f = 40
eyepiece.x = -20

#aperture = lensdraw.aperture('ap')
#aperture.diameter = 55
#aperture.x = 10

optics = lensdraw.opticalSystem()

optics.addElement(camlens)
optics.addElement(eyepiece)
optics.addElement(ccd)


#optics.addElement(aperture)

project = lensdraw.tracingProject(optics)

project.setObject(-40)

project.solveAll()
project.report()

project.setInputPlane(-100,relativeTo='firstElement')
project.setOutputPlane(200,relativeTo='auto')

#project.addTracesFillFirstElement(h='top', group=1)
#project.addTracesFillFirstElement(h='bottom',group=2)

project.addFullFieldChiefRay(group=1,rays=5)
project.addFullFieldChiefRay(group=2,rays=5, negative= True)

project.traceAll()

drawing = lensdraw.lensrender(name = "magnifier.svg", project = project, size=(800,600))
#drawing.setHorizontalScaleFromProject(project)
drawing.setHorizontalScale(-370,30)
drawing.setVerticalScale(50,fill = 0.4)
#project.setOutputPlane(200,relativeTo='auto')

drawing.drawAll(virtuals=False)

drawing.save_drawing()







