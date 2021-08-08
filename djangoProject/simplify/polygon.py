class polygon:
    edge_keys = [[0, 0], [0, 0], [0, 0]]
    vertices = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    vertices_index = [0, 0, 0]
    normal = [0]
    area = 0
    material_index = 0

    def __init__(self, face, vertices, normal, vertices_index, area, material_index):
        self.edge_keys = [[0, 0], [0, 0], [0, 0]]
        self.vertices = [[0, 0, 0],[0, 0, 0],[0, 0, 0]]
        self.vertices_index = [0, 0, 0]
        self.normal = [0]
        self.area = 0
        self.material_index = 0
        # self.edge_keys[0] = (face[0], face[1])
        # self.edge_keys[1] = (face[0], face[2])
        # self.edge_keys[2] = (face[1], face[2])

    def set_edge_keys(self, face):
        self.edge_keys[0] = (face[0], face[1])
        self.edge_keys[1] = (face[0], face[2])
        self.edge_keys[2] = (face[1], face[2])

    def set_normal(self, normal):
        self.normal = normal
        self.normal = tuple(self.normal)

    def set_vertices_index(self, vertices_index):
        self.vertices_index = vertices_index

    def set_vertices(self, vertices):
        self.vertices[0] = vertices[self.vertices_index[0]]
        self.vertices[1] = vertices[self.vertices_index[1]]
        self.vertices[2] = vertices[self.vertices_index[2]]

    def set_area(self, area):
        self.area = area

