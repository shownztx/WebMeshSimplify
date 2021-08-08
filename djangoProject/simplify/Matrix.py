import numpy as np
from test import Vector


def MulPosition(a, b): # (a Matrix)     b (Vector)
    x = a[0,0]*b.X + a[0,1]*b.Y + a[0,2]*b.Z + a[0,3]
    y = a[1,0]*b.X + a[1,1]*b.Y + a[1,2]*b.Z + a[1,3]
    z = a[2,0]*b.X + a[2,1]*b.Y + a[2,2]*b.Z + a[2,3]
    return Vector.Vector(x, y, z)


def QuadricVector(a): # (a Matrix)
    b = np.array([[a[0,0], a[0,1], a[0,2], a[0,3]],
		        [a[1,0], a[1,1], a[1,2], a[1,3]],
		        [a[2,0], a[2,1], a[2,2], a[2,3]],
		        [0, 0, 0, 1]])
    return MulPosition(np.linalg.inv(b), Vector.Vector())

def QuadricError(a, v):
	return (v.X*a[0,0]*v.X + v.Y*a[1,0]*v.X + v.Z*a[2,0]*v.X + a[3,0]*v.X +
		v.X*a[0,1]*v.Y + v.Y*a[1,1]*v.Y + v.Z*a[2,1]*v.Y + a[3,1]*v.Y +
		v.X*a[0,2]*v.Z + v.Y*a[1,2]*v.Z + v.Z*a[2,2]*v.Z + a[3,2]*v.Z +
		v.X*a[0,3] + v.Y*a[1,3] + v.Z*a[2,3] + a[3,3])