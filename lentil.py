import lensdraw
import numpy as np
import yaml
import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('optics', help="YAML file containing optical set up specification")
parser.add_argument('-t','--trace', default="chief-marginals-9", help="YAML file containing tracing project specification")
parser.add_argument('-o','--object', type=float, help="Object distance. Overrides any object distance specified in trace file")

args = parser.parse_args()

opticspec = open(args.optics+".yaml",'r')
opticdic = yaml.load(opticspec,Loader=yaml.Loader)

optics = lensdraw.opticalSystem()
optics.loadElements(opticdic["optics"])

#optics.printSystem()

project = lensdraw.tracingProject(optics)

with open(args.trace + ".yaml",'r') as tracespec:
	tracedic = yaml.load(tracespec,Loader=yaml.Loader)

object_distance = -20

trace_object = tracedic.get('object',None)

if trace_object is not None:

	object_distance = trace_object.get('x',object_distance)

if args.object is not None:

	object_distance = -abs(args.object)




#print(tracedic)

#project.setObject(tracedic['object']['x'],tracedic['object']['d'])

project.setObject(object_distance,15)

#project.setObject(-270,size=10)

project.solveAll()

original_stdout = sys.stdout # Save a reference to the original standard output

with open(args.optics+".txt", 'w') as f:
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

drawing = lensdraw.lensrender(name = args.optics+".svg", project = project, size=(800,600))
drawing.setHorizontalScale(-80,200)
drawing.setVerticalScale(50,fill = 0.3)

drawing.element_thickness = 10
drawing.scale_curvature = 10

drawing.drawAll(virtuals=True)

drawing.save_drawing()








