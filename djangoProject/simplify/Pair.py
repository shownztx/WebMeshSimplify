from test import Matrix
from test import Vector
import numpy as np

class Pair:
    # A, B        *Vertex
    def __init__(self, A, B, Index=-1, Removed=False, CachedError=-1):
        if B.Vector.Less(A.Vector):
            self.A = B
            self.B = A
            self.Index = Index
            self.Removed = Removed
            self.CachedError = CachedError
        else:
            self.A = A
            self.B = B
            self.Index = Index
            self.Removed = Removed
            self.CachedError = CachedError

    def Return_AB(self):
        return(self.A.Return_XYZ(),self.B.Return_XYZ())   

    def Return_ABQ(self):
        return(self.A.Return_Q(),self.B.Return_Q())  

    def Quadric(self):
        return self.A.Quadric + self.B.Quadric

    def Vector(self):
        q = self.Quadric()
        if np.absolute(np.linalg.det(q)) > 1e-3:
            v = Matrix.QuadricVector(q)
            if not( np.isnan(v.X) or np.isnan(v.X) or np.isnan(v.X) ):
                return v
        n = 32
        a = self.A.Vector
        b = self.B.Vector
        d = b.Sub(a)
        bestE = -1.0
        bestV = {}

        for i in range(n):
            t =  ( i / n )
            v = a.Add(d.MulScalar(t))
            e = Matrix.QuadricError(q, v)
            if bestE < 0 or e < bestE :
                bestE = e
                bestV = v
        return bestV

    def getLayerNorm(self, NormalNorm):
        if NormalNorm > 0.8 and NormalNorm <= 1:
            NormalNorm = 1
        if NormalNorm > 0.6 and NormalNorm <= 0.8:
            NormalNorm = 0.9             
        if NormalNorm > 0.4 and NormalNorm <= 0.6:
            NormalNorm = 0.4
        if NormalNorm > 0.2 and NormalNorm <= 0.4:
            NormalNorm = 0.2
        if NormalNorm >= 0 and NormalNorm <= 0.2:
            NormalNorm = 0.1
        if NormalNorm < 0:
            NormalNorm = 0.01
        return NormalNorm

    def Error(self):
        if self.CachedError < 0 :
            # print("p.A:",self.A.NormalNorm,"p.B:",self.B.NormalNorm)
            # self.CachedError = Matrix.QuadricError(self.Quadric(), self.Vector()) \
            #     * self.getLayerNorm(self.A.NormalNorm) * self.getLayerNorm(self.B.NormalNorm)

            self.CachedError = Matrix.QuadricError(self.Quadric(), self.Vector()) \
                * (self.A.NormalNorm) * (self.B.NormalNorm)            
            # print("p.A:",self.getLayerNorm(self.A.NormalNorm),"p.B:",self.getLayerNorm(self.B.NormalNorm))
        return self.CachedError

