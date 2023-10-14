# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:12:32 2023

@author: jhoyland
"""


import svgwrite
from svgwrite.extensions import Inkscape
import numpy as np

number = 72

dwg = svgwrite.Drawing('svgwrite-example.svg', profile='tiny')



bnum = format(108,"08b")

print(bnum)

dx = 0
for c in bnum:
    dx += 10
    dwg.add(dwg.text(c,dx,0))
    
    
dwg.save()