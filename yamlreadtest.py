import lensdraw
import numpy as np
import yaml
import sys

name = "PedrottiCh2Q23c"


opticspec = open(name+".yaml",'r')
opticdic  = yaml.load(opticspec,Loader=yaml.Loader)

optics = lensdraw.opticalSystem()
optics.loadElements(opticdic["optics"])

#optics.printSystem()

project = lensdraw.tracingProject(optics)

tracespec = open("chief-marginals-9.yaml",'r')
tracedic = yaml.load(tracespec,Loader=yaml.Loader)

#print(tracedic)

#project.setObject(tracedic['object']['x'],tracedic['object']['d'])

project.setObject(-20,10)

#project.setObject(-270,size=10)

project.solveAll()

original_stdout = sys.stdout # Save a reference to the original standard output

with open(name+".txt", 'w') as f:
	sys.stdout = f # Change the standard output to the file we created.
	project.report()
	sys.stdout = original_stdout # Re


project.setInputPlane(**tracedic['inputPlane'])
project.setOutputPlane(**tracedic['outputPlane'])

for rg in tracedic['rays']:

	typ = rg.pop('type')

	if typ == 'chief':

		project.addChiefRays(**rg)

	if typ == 'marginal':

		project.addMarginalRay(**rg)

	if typ == 'single':

		project.addTrace(**rg)


#project.setInputPlane(-100,relativeTo='object')
#project.setOutputPlane(100,relativeTo='image')

#project.addChiefRays(h=1,method="object",group=1,rays=17)
#project.addChiefRays(h=-1,method="object",group=2,rays=17)



project.traceAll()

drawing = lensdraw.lensrender(name = name+".svg", project = project, size=(800,600))
drawing.setHorizontalScale(-80,200)
drawing.setVerticalScale(50,fill = 0.3)

drawing.element_thickness = 10
drawing.scale_curvature = 10

drawing.drawAll(virtuals=True)

drawing.save_drawing()








