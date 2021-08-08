import copy

from djangoProject.simplify.polygon import polygon


class polygons:
    # instancelist = [polygon() for i in range(29)]
    _count = -1
    face_length = 0
    vertice_length = 0
    polygons = []

    def __init__(self, faces, vertices, normals, area):
        self.face_length = len(faces)
        self.vertice_length = len(vertices)
        for i in range(self.face_length):
            self.polygons.append(copy.deepcopy(polygon([0, 0, 0], [0, 0, 0], [0], [0, 0, 0], 0, 0)))

        for i in range(self.face_length):
            self.polygons[i].set_edge_keys(faces[i])
            self.polygons[i].set_normal(normals[i])
            self.polygons[i].set_vertices_index(faces[i])
            self.polygons[i].set_vertices(vertices)
            self.polygons[i].set_area(area[i])

    def __iter__(self):
        return self

    def __next__(self):
        self._count = self._count + 1
        if self._count < self.face_length:
            return self.polygons[self._count]
        else:
            raise StopIteration()

    def __getitem__(self, n):
        return self.polygons[n]