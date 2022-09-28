# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 21:58:06 2022

@author: hoyla
"""

import _old_lensdraw as lens
import svgwrite
from svgwrite.extensions import Inkscape
from IPython.display import SVG, display


dwg = svgwrite.Drawing('lens-eg.svg',size=(640,480),profile='tiny')
inkscape = Inkscape(dwg)


optics = lens.opticalSystem()

optics.addLens("L1",0,15,25)
optics.addLens("L2",20,30,50)

project = lens.tracingProject(optics)

project.setObject(-30)
project.setInputPlane()

project.solveAll()

project.addTracesFillFirstElement("top")

project.report()

optics.draw(dwg)

dwg.save()

display(SVG(dwg.tostring()))
