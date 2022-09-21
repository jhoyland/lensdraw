# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 10:52:50 2022

@author: hoyla
"""

class foo:
    
    str = "Hello from foo"
    
    prinfunc = None
    
    def __init__(self,x):
        
        self.x = x
        
    def printme(self):
        
        foo.prinfunc(self)
        
class bar(foo):
    
    str = "Hello from bar"
    
    def __init__(self,x,y):
        
        super().__init__(x)
        
        self.y = y
        

def printfoo(f):
    
    print(f.str)
    print(f.x)


def printbar(b):

    print(b.str)
    print(b.x)
    print(b.y)
    
    
f1 = foo(1)
f2 = foo(2)

b1 = bar(5,8)

printfoo(f1)

printbar(b1)
printfoo(b1)


foo.printfunc = printfoo

f1.printme()
        
        