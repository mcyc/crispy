"""
A place host specialized binary strucutres

"""

import numpy as np

#=====================================================================
# 2D structures

def eight_con():
    return np.ones((3, 3))

# Create 4 to 8-connected elements to use with binary hit-or-miss
struct1 = np.array([[1, 0, 0],
                    [0, 1, 1],
                    [0, 0, 0]])

struct2 = np.array([[0, 0, 1],
                    [1, 1, 0],
                    [0, 0, 0]])

# Next check the three elements which will be double counted
check1 = np.array([[1, 1, 0, 0],
                   [0, 0, 1, 1]])

check2 = np.array([[0, 0, 1, 1],
                   [1, 1, 0, 0]])

check3 = np.array([[1, 1, 0],
                   [0, 0, 1],
                   [0, 0, 1]])