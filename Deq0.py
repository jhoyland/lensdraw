import lensdraw
import numpy as np

optics = lensdraw.opticalSystem()

optics.addLens('lens1',0,10,25,form='bi')
optics.addLens('lens2',46,16,35,afterElement='lens1',form='bi',flatLeft=True)


project = lensdraw.tracingProject(optics)

project.setObject(-15,size=5)

project.solveAll()
project.report()

project.setInputPlane(-40,relativeTo='object')
project.setOutputPlane(80,relativeTo='auto')

#project.addChiefRays(h=1,method="object",group=1,rays=17)
#project.addChiefRays(h=-1,method="object",group=2,rays=17)

heights = np.linspace(-6,8,8)

#project.addTracesHeightRange(heights,-0.2,group=1)
#project.addTracesHeightRange(-1*heights,0.2,group=2)

angles = np.linspace(-0.3,0.3,8)

project.addTracesAngleRange('top',angles,group=1)
project.addTracesAngleRange('bottom',-1*angles,group=2)

#.2,group=0)
project.traceAll()

drawing = lensdraw.lensrender(name = "Deq0.svg", project = project, size=(600,600))
drawing.setHorizontalScale(-60,110)
drawing.setVerticalScale(30,fill = 0.3)
drawing.scale_curvature = 10
drawing.element_thickness = 16

drawing.drawAll(virtuals=False)

drawing.save_drawing()







