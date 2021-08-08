class Face:

    # V1, V2, V3 *Vertex
    def __init__(self, V1, V2, V3, Removed = False, MaterialIndex = -1):
        self.V1 = V1
        self.V2 = V2
        self.V3 = V3
        self.Removed = Removed
        self.MaterialIndex = MaterialIndex
    
    def __hash__(self):
        return hash(str(self.V1.Return_XYZ())+ str(self.V2.Return_XYZ()) + str(self.V3.Return_XYZ()))

    def __eq__(self, other):
        if self.V1.Vector == other.V1.Vector and self.V2.Vector == other.V2.Vector and self.V3.Vector == other.V3.Vector:
            return True
        return False

    def Return_V1V2V3(self):
        return(self.V1.Return_XYZ(),self.V2.Return_XYZ(),self.V3.Return_XYZ())   

    def Return_V1V2V3Q(self):
        return(self.V1.Return_Q(),self.V2.Return_Q(),self.V3.Return_Q())  

    def Degenerate(self):
        v1 = self.V1.Vector
        v2 = self.V2.Vector
        v3 = self.V3.Vector
        return ((v1 == v2) or (v1 == v3) or (v2 == v3))
    
    def Normal(self):
        e1 = self.V2.Vector.Sub(self.V1.Vector)
        e2 = self.V3.Vector.Sub(self.V1.Vector)
        return e1.Cross(e2).Normalize()