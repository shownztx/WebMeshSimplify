import os
import sys
import math
import json
import logging
import datetime
import mathutils
import collections
import numpy as np
import scipy.sparse
import scipy.linalg
import scipy.cluster
import scipy.sparse.linalg
import scipy.sparse.csgraph
from polygons import polygons

from test import Face
from test import Pair
# from test import Listh
from test import Vertex
from test import Vector
# from test import Heapgo
from test import PairKey
from test import Triangle


def normalization(data, maxData, minData):
    dis = maxData - minData
    print("data - minData: ", data - minData, "    dis: ", dis)
    try:
        ans = (data - minData) / dis
    except:
        print("data - minData: ", data - minData, "    dis: ", dis)
    return ans


def getvectorVertexNormalNorm(vertexFaces, materialIndexNorm, triangles, path):
    # path = 'testvectorNorm.json'
    for key, value in materialIndexNorm.items():
        print(key, "--", value)

    # vertexFaces key:vertex value:faces list，是一对多的map表
    vectorVertexNormalNorm = {}
    vectorVertex = {}
    vectorNorm = {}

    for vertex, faces in vertexFaces.items():
        maxNorm = 0
        for face in faces:
            norm = materialIndexNorm[face.MaterialIndex]
            if norm > maxNorm:
                maxNorm = norm
        # 获取到法向量指标
        vectorNorm[vertex.Return_XYZ_json()] = maxNorm

    with open(path, 'w') as f:
        json.dump(vectorNorm, f)
    return vectorNorm


def show_material(mesh):
    polygons = mesh.polygons
    vertices = mesh.vertices
    materialPolygons = collections.defaultdict(list)
    materialPolygonNormal = collections.defaultdict(list)
    materialNorm = {}

    materialNums = 0

    for key, value in polygons.items():
        materialPolygons[value.material_index].append(value)
        if value.material_index > materialNums:
            materialNums = value.material_index

    materialNums += 1  # 因为是数目，所以要加1

    for key, value in materialPolygons.items():
        # print(key, " ", "type(value):", type(value), " ", (value))
        for polygon in value:
            # print(type(polygon.vertices))
            # print(polygon.vertices)
            # print(type(polygon.normal))
            normal = np.array([float(polygon.normal[0]), float(polygon.normal[1]), float(polygon.normal[2])])
            area = np.array([polygon.area])
            normal_area = np.concatenate((normal, area), axis=0)  # 前三个是normal 后一个是area
            # print(normal_area)
            # print(normal_area[0])
            # print(normal_area[1])
            materialPolygonNormal[polygon.material_index].append(normal_area)

    minNorm = 999
    for i in range(materialNums):
        normalsProduct = np.ones(3)
        # for normal_area in materialPolygonNormal[i]:
        #     normalsProduct = np.cross(normalsProduct, normal_area[:3]) # 前三个是normal 
        #     normalsProduct = normalsProduct * normal_area[3] # 第四个是Area

        # 长度不是4的倍数的要补0
        # if not len(materialPolygonNormal[i])%4  == 0:
        #     # print(len(materialPolygonNormal[i])%4)
        #     for k in range((4 - len(materialPolygonNormal[k])%4)):
        #         materialPolygonNormal[k].append(0)

        for j in range(0, len(materialPolygonNormal[i]), 25):
            # for j in range(0, len(materialPolygonNormal[i]), 16):
            normalsProduct = np.cross(normalsProduct, materialPolygonNormal[i][j][:3])  # 前三个是normal
            normalsProduct = normalsProduct * materialPolygonNormal[i][j][3]  # 第四个是Area
        n = np.linalg.norm(normalsProduct)
        if n < minNorm and n > 0:
            minNorm = n
        if n == 0:
            n = minNorm
        print("nums: ", i, "   norm(normalsProduct): ", n)
        materialNorm[i] = - np.log10(n)
        # materialNorm[i] = (np.linalg.norm(normalsProduct))

    maxNorm = max(materialNorm.values())
    minNorm = min(materialNorm.values())

    for key, value in materialNorm.items():
        '''
        key 为材质标签， value为曲折度量指标 Products为乘积 Norm为取模之后的数值
        shawn:  0 64.0
        shawn:  1 64.0
        '''
        print("key: ", key, "   value: ", value)
        if value == 0:
            value = 1
        new_value = normalization(value, maxNorm, minNorm)
        materialNorm[key] = new_value

    return materialNorm


def save_to_json(mesh, path):
    from polygons import polygons
    faces = polygons(mesh.faces, mesh.vertices, mesh.face_normals)
    polygons = faces.polygons
    vertices = mesh.vertices
    vectorVertex = {}
    indexVector = {}
    vertexFaces = collections.defaultdict(list)

    for index, vertice in enumerate(vertices):
        vec = Vector.Vector(vertice[0], vertice[1], vertice[2])
        indexVector[index] = vec
        vectorVertex[vec] = Vertex.Vertex(vec)

    triangles = [] * len(polygons)
    for index, polygon in enumerate(polygons):
        print(polygon.vertices[0], polygon.vertices[1], polygon.vertices[2])
        triangles.append(Triangle.Triangle(indexVector[polygon.vertices[0]],
                                           indexVector[polygon.vertices[1]],
                                           indexVector[polygon.vertices[2]],
                                           polygon.material_index)
                         )

    for triangle in triangles:
        v1 = vectorVertex[triangle.V1]
        v2 = vectorVertex[triangle.V2]
        v3 = vectorVertex[triangle.V3]
        material_index = triangle.MaterialIndex
        # print("material_index before: ",material_index)

        face = Face.Face(v1, v2, v3, False, material_index)
        # 把每个面都遍历了
        vertexFaces[v1].append(face)
        vertexFaces[v2].append(face)
        vertexFaces[v3].append(face)

    materialIndexNorm = show_material(mesh)
    vectorVertexNormalNorm = getvectorVertexNormalNorm(vertexFaces, materialIndexNorm, triangles, path)
    print("-------------save json to " + path + "-------------")
