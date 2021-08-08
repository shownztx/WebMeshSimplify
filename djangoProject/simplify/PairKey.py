class PairKey:

    def __init__(self, A, B):
        if B.Vector.Less(A.Vector):
            self.A = B.Vector
            self.B = A.Vector
        else:
            self.A = A.Vector
            self.B = B.Vector
    
    def __hash__(self):
        return hash(str(self.A.Return_XYZ()) + str(self.B.Return_XYZ()))

    def __eq__(self, other):
        if self.A == other.A and self.B == other.B:
            return True
        return False

    def Print(self):
        print("{", self.A.X, self.A.Y, self.A.Z,"}", "{", self.B.X, self.B.Y, self.B.Z,"}")

    def Return_XYZ(self):
        return(self.A.X, self.A.Y, self.A.Z, self.B.X, self.B.Y, self.B.Z)