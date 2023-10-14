import lensdraw
import numpy as np

optics = lensdraw.opticalSystem()

optics.addLens('objective',0,150,50)
optics.addLens('eyepiece',240,-210,50,afterElement='objective')

project = lensdraw.tracingProject(optics)

project.setObject(-270,size=10)

project.solveAll()
project.report()

project.setInputPlane(-100,relativeTo='object')
project.setOutputPlane(100,relativeTo='image')

project.addChiefRays(h=1,method="object",group=1,rays=17)
project.addChiefRays(h=-1,method="object",group=2,rays=17)



project.traceAll()

drawing = lensdraw.lensrender(name = "lens2.svg", project = project, size=(800,600))
drawing.setHorizontalScale(-350,450)
drawing.setVerticalScale(50,fill = 0.3)

drawing.element_thickness = 20

drawing.drawAll(virtuals=True)

drawing.save_drawing()







