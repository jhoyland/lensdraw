import lensdraw
import numpy as np

optics = lensdraw.opticalSystem()

optics.addLens('lens1',0,51.5,50,form='plano')
optics.addLens('lens2',15,30.7,40,afterElement='lens1',form='plano')
optics.addLens('lens3',35,18.5,27,afterElement='lens2',form='plano',flatLeft=True)
optics.addLens('lens4',35,33.2,55,afterElement='lens3',form='plano',flatLeft=True)


project = lensdraw.tracingProject(optics)

project.setObject(-np.inf,size=10)

project.solveAll()
project.report()

project.setInputPlane(-25.4,relativeTo='firstElement')
project.setOutputPlane(83.9,relativeTo='lastElement')

#project.addChiefRays(h=1,method="object",group=1,rays=17)
#project.addChiefRays(h=-1,method="object",group=2,rays=17)

heights = np.linspace(-4,10,8)

#project.addTracesHeightRange(heights,-0.15,group=1)
#project.addTracesHeightRange(-1*heights,0.15,group=2)

project.addTrace(4,-0.15,group=1)

#.2,group=0)
project.traceAll()

drawing = lensdraw.lensrender(name = "lens4single.svg", project = project, size=(600,600))
drawing.setHorizontalScale(-30,150)
drawing.setVerticalScale(30,fill = 0.3)
drawing.scale_curvature = 10
drawing.element_thickness = 17

drawing.drawAll(virtuals=False)

drawing.save_drawing()







