import collections
import json
import math
# Controls weight of geodesic to angular distance. Values closer to 0 give
# the angular distance more importance, values closer to 1 give the geodesic
# distance more importance
import os

from djangoProject.simplify import subprocess
import time

import numpy
import scipy
import scipy.cluster
import scipy.linalg
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
import trimesh

from djangoProject.simplify.Face import Face
from djangoProject.simplify.Triangle import Triangle
from djangoProject.simplify.Vector import Vector
from djangoProject.simplify.Vertex import Vertex
from djangoProject.simplify.polygons import polygons

delta = 0.03
eta = 0.15


def _list_plus(list1, list2):
    if len(list1) != len(list2):
        return False
    ans = [0, 0, 0]
    for i in range(3):
        ans[0] = list1[0] + list2[0]
        ans[1] = list1[1] + list2[1]
        ans[2] = list1[2] + list2[2]
    return ans


def _list_minus(list1, list2):
    if len(list1) != len(list2):
        return False
    ans = [0, 0, 0]
    for i in range(3):
        ans[0] = list1[0] - list2[0]
        ans[1] = list1[1] - list2[1]
        ans[2] = list1[2] - list2[2]
    return ans


def _list_multiplication(list1, list2):
    a1 = list1[0] * list2[0]
    a2 = list1[1] * list2[1]
    a3 = list1[2] * list2[2]
    return a1 + a2 + a3


def _list_cos(list1, list2):
    if len(list1) != len(list2):
        return False
    ans = _list_multiplication(list1, list2) / (_list_length(list1) * _list_length(list2))
    return ans


def _face_center(mesh, face):
    """Computes the coordinates of the center of the given face"""
    center = [0, 0, 0]
    for vert in face.vertices:
        center = _list_plus(vert, center)
    new_list = [x / len(face.vertices) for x in center]
    return new_list


def _list_length(list1):
    if (len(list1) != 3):
        return False
    ans = math.sqrt(math.pow(list1[0], 2) + math.pow(list1[1], 2) + math.pow(list1[2], 2))
    return ans


def _geodesic_distance(mesh, face1, face2, edge):
    """Computes the geodesic distance over the given edge between
    the two adjacent faces face1 and face2"""
    edge_center = (mesh.vertices[edge[0]] + mesh.vertices[edge[1]]) / 2
    return _list_length(_list_minus(edge_center, _face_center(mesh, face1))) + \
           _list_length(_list_minus(edge_center, _face_center(mesh, face2)))


def _angular_distance(mesh, face1, face2):  # 其实是余弦距离
    """Computes the angular distance of the given adjacent faces"""
    angular_distance = (1 - _list_cos(face1.normal, face2.normal))
    if _list_multiplication(face1.normal, (_list_minus(_face_center(mesh, face2), _face_center(mesh, face1)))) < 0:
        # convex angles are not that bad so scale down distance a bit
        # 凸角不是那么糟糕，所以要把距离缩小一些。
        angular_distance *= eta
    return angular_distance


def _create_distance_matrix(mesh):
    """Creates the matrix of the angular and geodesic distances
    between all adjacent faces. The i,j-th entry of the returned
    matrices contains the distance between the i-th and j-th face.
    """
    l = len(mesh.faces)

    faces = polygons(mesh.faces, mesh.vertices, mesh.face_normals, mesh.area_faces)
    # map from edge-key to adjacent faces
    adj_faces_map = {}
    # find adjacent faces by iterating edges
    for index, face in enumerate(faces):
        for edge in face.edge_keys:
            if (edge[0] > edge[1]):
                new_edge = (edge[1], edge[0])
            else:
                new_edge = (edge[0], edge[1])
            if new_edge in adj_faces_map:
                adj_faces_map[new_edge].append(index)  # 一对多
            else:
                adj_faces_map[new_edge] = [index]

    # helping vectors to create sparse matrix later on
    row_indices = []
    col_indices = []
    Gval = []  # values for matrix of angular distances
    Aval = []  # values for matrix of geodesic distances
    # iterate adjacent faces and calculate distances
    for edge, adj_faces in adj_faces_map.items():
        if len(adj_faces) == 2:
            i = adj_faces[0]
            j = adj_faces[1]
            # 一条边连接的两个面
            Gtemp = _geodesic_distance(mesh, faces[i], faces[j], edge)  # 测地距离
            Atemp = _angular_distance(mesh, faces[i], faces[j])  # 角距离  # 其实是余弦距离
            Gval.append(Gtemp)
            Aval.append(Atemp)
            row_indices.append(i)
            col_indices.append(j)
            # add symmetric entry
            Gval.append(Gtemp)
            Aval.append(Atemp)
            row_indices.append(j)
            col_indices.append(i)

        elif len(adj_faces) > 2:
            print("Edge with more than 2 adjacent faces: " + str(adj_faces) + "!")

    Gval = numpy.array(Gval)
    Aval = numpy.array(Aval)
    # delta是去全局变量，外部传入的
    values = delta * Gval / numpy.mean(Gval) + \
             (1.0 - delta) * Aval / numpy.mean(Aval)

    # create sparse matrix
    distance_matrix = scipy.sparse.csr_matrix(
        (values, (row_indices, col_indices)), shape=(l, l))
    return distance_matrix


def _create_affinity_matrix(mesh):
    """Create the adjacency matrix of the given mesh"""

    l = len(mesh.faces)
    print("mesh_segmentation: Creating distance matrices...")
    distance_matrix = _create_distance_matrix(mesh)

    print("mesh_segmentation: Finding shortest paths between all faces...")
    # for each non adjacent pair of faces find shortest path of adjacent faces
    W = scipy.sparse.csgraph.dijkstra(distance_matrix)
    inf_indices = numpy.where(numpy.isinf(W))  # 不可达的
    W[inf_indices] = 0  # 标记不可达的

    print("mesh_segmentation: Creating affinity matrix...")
    # change distance entries to similarities
    sigma = W.sum() / (l ** 2)  # 平方
    den = 2 * (sigma ** 2)
    W = numpy.exp(-W / den)
    W[inf_indices] = 0
    numpy.fill_diagonal(W, 1)  # 充任意维度的数组的主对角线。

    return W


def _initial_guess(Q, k):
    """Computes an initial guess for the cluster-centers

    Chooses indices of the observations with the least association to each
    other in a greedy manner. Q is the association matrix of the observations.
    """

    # choose the pair of indices with the lowest association to each other
    # 选择彼此关联度最低的一对指数
    min_indices = numpy.unravel_index(numpy.argmin(Q), Q.shape)
    # print("-----------min_indices------------")
    # print(min_indices)
    chosen = [min_indices[0], min_indices[1]]
    # print("-----------chosen------------")
    # print(chosen)
    for _ in range(2, k):
        # Take the maximum of the associations to the already chosen indices for
        # every index. The index with the lowest result in that therefore is the
        # least similar to the already chosen pivots so we take it.
        # Note that we will never get an index that was already chosen because
        # an index always has the highest possible association 1.0 to itself
        '''
        取每个指数与已经选择的指数的关联度的最大值。因此，其中结果最低的指数与已经选择的枢轴的相似度最低，
        所以我们取它。 请注意，我们永远不会得到一个已经被选中的指数，因为一个指数总是与自己有最高的可能关联1.0。
        '''
        # print("-----------chosen------------")
        # print(chosen)
        # print("-----------Q[chosen,:]------------")
        # print(Q[chosen, :])
        # print("-----------numpy.max(Q[chosen,:])------------")
        # print(numpy.max(Q[chosen, :]))
        new_index = numpy.argmin(numpy.max(Q[chosen, :], axis=0))
        chosen.append(new_index)
    #     print("-----------new_index------------")
    #     print(new_index)
    # print("-----------chosen set------------")
    # print(chosen)
    return chosen


def segment_mesh(mesh, k, coefficients, action, ev_method, kmeans_init):
    """Segments the given mesh into k clusters and performs the given
    action for each cluster
    """

    # set coefficients
    global delta
    global eta
    delta = coefficients[0]
    eta = coefficients[1]
    ev_method = 'sparse'
    kmeans_init = 'Liu'

    # affinity matrix
    W = _create_affinity_matrix(mesh)
    # print("---------------W:---------------")
    # print(W)
    print("mesh_segmentation: Calculating graph laplacian...")
    # degree matrix
    Dsqrt = numpy.sqrt(numpy.reciprocal(W.sum(1)))  # 每一行的和，取倒数在开根号。
    # print("---------------Dsqrt:---------------")
    # print(Dsqrt)
    # graph laplacian
    L = ((W * Dsqrt).transpose() * Dsqrt).transpose()
    # print("---------------L:---------------")
    # print(L)
    print("mesh_segmentation: Calculating eigenvectors...")
    # get eigenvectors
    if ev_method == 'dense':
        _, V = scipy.linalg.eigh(L, eigvals=(L.shape[0] - k, L.shape[0] - 1))
        # print("L.shape[0] - k: ", L.shape[0] - k, ", L.shape[0] - 1: ", L.shape[0] - 1)
        # print("---------------V:---------------")
        # print(V)

    else:
        # 默认
        k = int(k)
        E, V = scipy.sparse.linalg.eigsh(L, k)
        # print("---------------V:---------------")
        # print(V)
        # print("---------------E:---------------")
        # print(E)
        # normalize each row to unit length
    V /= numpy.linalg.norm(V, axis=1)[:, None]
    # print("---------------norm V:---------------")
    # print(V)
    start = time.time()
    if kmeans_init == 'kmeans++':
        print("mesh_segmentation: Applying kmeans...")
        _, idx = scipy.cluster.vq.kmeans2(V, k, minit='++', iter=50)
    else:
        print("mesh_segmentation: Preparing kmeans...")
        # compute association matrix

        Q = V.dot(V.transpose())
        # print("---------------Q:---------------")
        # print(Q)
        # compute initial guess for clustering
        initial_centroids = _initial_guess(Q, k)

        print("mesh_segmentation: Applying kmeans...")
        # print("-----------V[initial_centroids,:]-------------")
        # print(V[initial_centroids, :])
        _, idx = scipy.cluster.vq.kmeans2(V, V[initial_centroids, :], iter=50)
    end = time.time()
    print("mesh_segmentation: Done clustering!: ", end - start)
    # perform action with the clustering result
    print("---------------idx:---------------")
    print(idx)

    # 为每个面的材质索引赋值，就是类别
    faces = polygons(mesh.faces, mesh.vertices, mesh.face_normals, mesh.area_faces)
    for i, id in enumerate(idx):
        faces.polygons[i].material_index = id
    # for i in range(len(mesh.faces)):
    #     print(faces.polygons[i].material_index)
    return faces


def normalization(data, maxData, minData):
    dis = maxData - minData
    print("data - minData: ", data - minData, "    dis: ", dis)
    try:
        ans = (data - minData) / dis
    except:
        print("data - minData: ", data - minData, "    dis: ", dis)
    return ans


def get_seg_index(facess, mesh):
    l = len(mesh.faces)
    polygons = facess.polygons
    vertices = mesh.vertices
    materialPolygons = collections.defaultdict(list)
    materialPolygonNormal = collections.defaultdict(list)
    materialNorm = {}
    materialNums = 0

    for i in range(l):
        # print('ztx: ', polygons[i].material_index)
        materialPolygons[polygons[i].material_index].append(polygons[i])
        if polygons[i].material_index > materialNums:
            materialNums = polygons[i].material_index
    materialNums += 1  # 因为是数目，所以要加1
    # print('ztx: ', materialPolygons[0])

    for key, value in materialPolygons.items():
        # print(key, " ", "type(value):", type(value), " ", (value))
        for polygon in value:
            # print('ztx polygon.normal: ', polygon.normal[0], polygon.normal[1], polygon.normal[2])
            normal = numpy.array([float(polygon.normal[0]), float(polygon.normal[1]), float(polygon.normal[2])])
            area = numpy.array([polygon.area])
            normal_area = numpy.concatenate((normal, area), axis=0)  # 前三个是normal 后一个是area
            materialPolygonNormal[polygon.material_index].append(normal_area)

    minNorm = 999
    # print('ztx materialNums: ', materialNums)
    for i in range(materialNums):
        normalsProduct = numpy.ones(3)
        for j in range(0, len(materialPolygonNormal[i]), 25):
            # for j in range(0, len(materialPolygonNormal[i]), 16):
            normalsProduct = numpy.cross(normalsProduct, materialPolygonNormal[i][j][:3])  # 前三个是normal
            normalsProduct = normalsProduct * materialPolygonNormal[i][j][3]  # 第四个是Area
        n = numpy.linalg.norm(normalsProduct)
        if n < minNorm and n > 0:
            minNorm = n
        if n == 0:
            n = minNorm
        print("nums: ", i, "   norm(normalsProduct): ", n)
        materialNorm[i] = - numpy.log10(n)
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


def getvectorVertexNormalNorm(vertexFaces, materialIndexNorm, path):
    # path = 'testvectorNorm.json'
    for key, value in materialIndexNorm.items():
        print(key, "--", value)
    # vertexFaces key:vertex value:faces list，是一对多的map表
    vectorNorm = {}

    for vertex, faces in vertexFaces.items():
        maxNorm = 0
        for face in faces:
            norm = materialIndexNorm[face.MaterialIndex]
            if norm > maxNorm:
                maxNorm = norm
        # 获取到法向量指标
        vectorNorm[vertex.Return_XYZ_json()] = maxNorm
    # "testvectorNorm.json"
    with open("testvectorNorm.json", 'w') as f:
        json.dump(vectorNorm, f)



def save_to_json(facess, mesh, materialIndexNorm, path):
    polygons = facess.polygons
    vertices = mesh.vertices
    vectorVertex = {}
    indexVector = {}
    vertexFaces = collections.defaultdict(list)

    for index, vertice in enumerate(vertices):
        vec = Vector(vertice[0], vertice[1], vertice[2])
        indexVector[index] = vec
        vectorVertex[vec] = Vertex(vec)

    triangles = [] * len(polygons)
    for index, polygon in enumerate(polygons):
        # print(polygon.vertices_index[0], polygon.vertices_index[1], polygon.vertices_index[2])
        triangles.append(Triangle(indexVector[polygon.vertices_index[0]],
                                  indexVector[polygon.vertices_index[1]],
                                  indexVector[polygon.vertices_index[2]],
                                  polygon.material_index)
                         )
    for triangle in triangles:
        v1 = vectorVertex[triangle.V1]
        v2 = vectorVertex[triangle.V2]
        v3 = vectorVertex[triangle.V3]
        material_index = triangle.MaterialIndex
        # print("material_index before: ",material_index)

        face = Face(v1, v2, v3, False, material_index)
        # 把每个面都遍历了
        vertexFaces[v1].append(face)
        vertexFaces[v2].append(face)
        vertexFaces[v3].append(face)
    getvectorVertexNormalNorm(vertexFaces, materialIndexNorm, path)
    print("-------------save json to " + path + "-------------")


def main(mesh, seg_factor, simp_factor):
    delta = 0.03
    eta = 0.15
    tmesh = trimesh.load('static/models/' + mesh)
    stl_origin_path = 'static/models/' + mesh
    stl_simp_path = 'static/models/' + str(mesh).split('.')[0] + '_simp.stl'
    faces = segment_mesh(tmesh, seg_factor, [delta, eta], '', '', '')
    # 保存法向量
    materialIndexNorm = get_seg_index(faces, tmesh)
    path = 'testvectorNorm.json'
    save_to_json(faces, tmesh, materialIndexNorm, path)
    # 执行简化
    parm = "-f " + str(simp_factor) + " " + str(stl_origin_path) + " " + str(
        stl_simp_path) + " " + "testvectorNorm.json"
    print(parm)
    time.sleep(5)
    while not os.path.exists(path):
        pass
    print("ztx: ", os.getcwd()) # 获取当前工作目录路径
    subprocess.Popen(["djangoProject/simplify/main.exe", "-f", str(simp_factor), str(stl_origin_path), str(stl_simp_path),
                      "testvectorNorm.json"])

    print('------over--------')
