import lensdraw
import numpy as np

optics = lensdraw.opticalSystem()

optics.addLens('lens1',0,35,50,form='bi')
optics.addLens('lens2',60,14,35,afterElement='lens1',form='bi',flatLeft=True)


project = lensdraw.tracingProject(optics)

project.setObject(-np.inf,size=10)

project.solveAll()
project.report()

project.setInputPlane(-25.4,relativeTo='firstElement')
project.setOutputPlane(0,relativeTo='image')

#project.addChiefRays(h=1,method="object",group=1,rays=17)
#project.addChiefRays(h=-1,method="object",group=2,rays=17)

heights = np.linspace(-4,10,8)

project.addTracesHeightRange(heights,-0.15,group=1)
project.addTracesHeightRange(-1*heights,0.15,group=2)

#.2,group=0)
project.traceAll()

drawing = lensdraw.lensrender(name = "Aeq0.svg", project = project, size=(600,600))
drawing.setHorizontalScale(-60,120)
drawing.setVerticalScale(30,fill = 0.3)
drawing.scale_curvature = 10
drawing.element_thickness = 16

drawing.drawAll(virtuals=False)

drawing.save_drawing()







