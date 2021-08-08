import math
import numpy as np
class Triangle:
    # v1: Vector
    def __init__(self, V1, V2, V3, MaterialIndex = -1):
        self.V1 = V1
        self.V2 = V2
        self.V3 = V3
        self.MaterialIndex = MaterialIndex
    
    def Quadric(self):
        n = self.Normal()
        x, y, z = self.V1.X, self.V1.Y, self.V1.Z
        a, b, c = n.X, n.Y, n.Z
        d = -a*x - b*y - c*z
        return np.array([a * a, a * b, a * c, a * d,
                        a * b, b * b, b * c, b * d,
                        a * c, b * c, c * c, c * d,
                        a * d, b * d, c * d, d * d,]).reshape(4, 4)
    
    def Normal(self):
        e1 = self.V2.Sub(self.V1)
        e2 = self.V3.Sub(self.V1)
        return e1.Cross(e2).Normalize()