import math
import numpy as np

class Vector:
    def __init__(self, X = 0, Y = 0, Z = 0):
        self.X = X
        self.Y = Y
        self.Z = Z

    def __hash__(self):
        return hash(str(self.Return_XYZ()))
        
    def __eq__(self, other):
        if self.X == other.X and self.Y == other.Y and self.Z == other.Z:
            return True
        return False

    def Print_XYZ(self):
        print("{",self.X," ",self.Y," ",self.Z ,"}")

    def Return_XYZ(self):
        return(self.X,self.Y,self.Z)   

    def Less(self, b):
        if self.X != b.X :
            return self.X < b.X

        if self.Y != b.Y:
            return self.Y < b.Y
        return self.Z < b.Z

    def Length(self):
        return math.sqrt(self.X * self.X + self.Y * self.Y + self.Z * self.Z)

    def Dot(self, b):
        return self.X*b.X + self.Y*b.Y + self.Z*b.Z

    def Cross(self, b):
        x = self.Y * b.Z - self.Z * b.Y
        y = self.Z * b.X - self.X * b.Z
        z = self.X * b.Y - self.Y * b.X
        return Vector(x, y, z)


    def Normalize(self):
        d = self.Length()
        return Vector(self.X / d, self.Y / d, self.Z / d)

    def Add(self, b):
        return Vector(self.X + b.X, self.Y + b.Y, self.Z + b.Z)

    def Sub(self, b):
        return Vector(self.X - b.X, self.Y - b.Y, self.Z - b.Z)

    def MulScalar(self, b):
        return Vector(self.X * b, self.Y * b, self.Z * b)
