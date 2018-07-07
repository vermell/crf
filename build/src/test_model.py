#!/usr/bin/env python2
import crf
import numpy as np

gm = crf.GraphicalModel(3,2)
for i in range(3):
    gm.addX(i, np.array([1,0,1]))
    gm.addY(i,1)
    gm.addUnary(i,i)

    gm.addX(i + 10, np.array([1,0,1]))
    gm.addY(i + 10,0)
    gm.addUnary(i + 10,i +10)
    gm.addX(i + 11, np.array([0,0,1]))
    gm.addY(i + 11,0)
    gm.addUnary(i + 11,i +11)
    

gm.print_info()

gm.learnModel()
