import lensdraw

lens = lensdraw.thinLens('lens')

lens.f = 50
lens.diameter = 50
lens.makeBi()

aperture = lensdraw.aperture('ap')
aperture.diameter = 55
aperture.x = 10

optics = lensdraw.opticalSystem()

optics.addElement(lens)
optics.addElement(aperture)

project = lensdraw.tracingProject(optics)

project.setObject(-65)

project.solveAll()
project.report()

project.setInputPlane(-100,relativeTo='firstElement')
project.setOutputPlane(200,relativeTo='auto')

project.addTracesFillFirstElement(h='top', group=1)
project.addTracesFillFirstElement(h='bottom',group=2)
project.traceAll()

drawing = lensdraw.lensrender(size=(800,600))
#drawing.setHorizontalScaleFromProject(project)
drawing.setHorizontalScale(-100,300)
drawing.setVerticalScale(50,fill = 0.2)
#project.setOutputPlane(200,relativeTo='auto')

drawing.drawOpticalSystem(optics)
drawing.drawSourceObject(project)
drawing.drawFinalImage(project)

for ray in project.traces:
	drawing.drawRayTrace(ray)

drawing.drawOpticalAxis()

drawing.drawOpticElement(project.entrancePupil, "pupils")
drawing.drawOpticElement(project.exitPupil, "pupils")
drawing.drawOpticElement(project.entranceWindow, "windows")
drawing.drawOpticElement(project.exitWindow, "windows")

drawing.save_drawing()







