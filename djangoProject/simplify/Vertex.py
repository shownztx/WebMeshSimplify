import numpy as np

class Vertex:
    def __init__(self, Vector, Quadric = np.zeros(16).reshape(4,4), NormalNorm = 1):
        # 设置缺省参数Quadric，格式为np.array
        self.Vector = Vector
        self.Quadric = Quadric
        self.NormalNorm = NormalNorm

    def __hash__(self):
        return hash(str(self.Return_XYZ()))

    def __eq__(self, other):
        if self.Vector.X == other.Vector.X and self.Vector.Y == other.Vector.Y and self.Vector.Z == other.Vector.Z:
            return True
        return False
    
    def Return_XYZ(self):
        return(self.Vector.X,self.Vector.Y,self.Vector.Z)   

    def Return_Q(self):
        return(self.Quadric)   
    
    def Return_XYZ_json(self):

        # return (str(format(self.Vector.X,'.6f')).replace('.', '_') \
        #     + " " + str(format(self.Vector.Y,'.6f')).replace('.', '_') \
        #     + " " + str(format(self.Vector.Z,'.6f')).replace('.', '_'))       

        return (str((self.Vector.X)).replace('.', '_') \
            + " " + str((self.Vector.Y)).replace('.', '_') \
            + " " + str((self.Vector.Z)).replace('.', '_'))               