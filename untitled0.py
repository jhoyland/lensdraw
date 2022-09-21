# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 22:13:45 2022

@author: hoyla
"""

import svgwrite
from IPython.display import SVG,display


dwg = svgwrite.Drawing('test1.svg',size=(640,480),profile='tiny')


pathstr = "M20,100 l80,0 l5,-80 l-85,0 z"

path = dwg.path(pathstr,fill="red")


pathstr2 = "M120,100 l80,0 l5,-80 l-85,0 z"

path2 = dwg.path(pathstr2,fill="blue")
path2.scale(2,2)

dwg.add(path)
dwg.add(path2)


display(SVG(dwg.tostring()))
