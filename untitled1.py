# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 09:06:26 2022

@author: hoyla
"""

class A:
    
    def __init__(self,a):
        
        self.a = a
        
    @classmethod 
    def from_class(cls,b):
        return cls(2*b.a)
    

one = A(2)
two = A.from_class(one)


print(one.a)
print(two.a)