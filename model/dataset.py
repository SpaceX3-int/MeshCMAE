# using the modelnet40 as the dataset, and using the processed feature matrixes
import json
import random
from pathlib import Path
import numpy as np
import os
import torch
import torch.utils.data as data
import trimesh
from scipy.spatial.transform import Rotation
import pygem
import pygem as pg
import copy
import csv


def randomize_mesh_orientation(mesh: trimesh.Trimesh):
    mesh1 = copy.deepcopy(mesh)
    axis_seq = ''.join(random.sample('xyz', 3))
    angles = [random.choice([0, 30, -30, 15, -15]) for _ in range(3)] # 45, 90, 120, 135, 180, 210, 225, 270, 300, 315
    rotation = Rotation.from_euler(axis_seq, angles, degrees=True)
    mesh1.vertices = rotation.apply(mesh1.vertices)
    return mesh1


def random_scale(mesh: trimesh.Trimesh):
    mesh.vertices = mesh.vertices * np.random.normal(1, 0.1, size=(1, 3))
    return mesh


def mesh_normalize(mesh: trimesh.Trimesh):
    mesh1 = copy.deepcopy(mesh)
    vertices = mesh1.vertices - mesh1.vertices.min(axis=0)
    vertices = vertices / vertices.max()
    mesh1.vertices = vertices
    return mesh1


def mesh_deformation(mesh: trimesh.Trimesh):
    ffd = pg.FFD([2, 2, 2])
    random = np.random.rand(6) * 0.1
    ffd.array_mu_x[1, 1, 1] = random[0]
    ffd.array_mu_y[1, 1, 1] = random[1]
    ffd.array_mu_z[1, 1, 1] = random[2]
    ffd.array_mu_x[0, 0, 0] = random[3]
    ffd.array_mu_y[0, 0, 0] = random[4]
    ffd.array_mu_z[0, 0, 0] = random[5]
    vertices = mesh.vertices
    new_vertices = ffd(vertices)
    mesh.vertices = new_vertices
    return mesh

def generate_face_labels_nearest(mesh, point_labels):
    '''Generate face labels using the nearest point algorithm'''
    face_labels = np.zeros(len(mesh.faces), dtype=int)
    for i, face in enumerate(mesh.faces):
        face_center = mesh.vertices[face].mean(axis=0)
        distances = np.linalg.norm(mesh.vertices - face_center, axis=1)
        nearest_vertex_index = np.argmin(distances)
        face_labels[i] = point_labels[nearest_vertex_index]
    return face_labels

# Example usage:
# mesh = trimesh.load_mesh('path/to/mesh.obj', process=False)
# point_labels = np.array([...])  # Point labels corresponding to mesh vertices
# face_labels = generate_face_labels_nearest(mesh, point_labels)

'''
def load_mesh(path, augments=[], request=[], seed=None):
    label = 0
    if 'guitar' in str(path):
        label = 15
    elif 'door' in str(path):
        label = 30
    elif 'radio' in str(path):
        label = 38
    elif 'curtain' in str(path):
        label = 13
    elif 'dresser' in str(path):
        label = 7
    elif 'bookshelf' in str(path):
        label = 9
    elif 'tent' in str(path):
        label = 21
    elif 'bottle' in str(path):
        label = 32
    elif 'lamp' in str(path):
        label = 23
    elif 'piano' in str(path):
        label = 5
    elif 'stool' in str(path):
        label = 16
    elif 'bench' in str(path):
        label = 28
    elif 'chair' in str(path):
        label = 37
    elif 'bathtub' in str(path):
        label = 10
    elif 'vase' in str(path):
        label = 33
    elif 'flower' in str(path):
        label = 31
    elif 'plant' in str(path):
        label = 34
    elif 'keyboard' in str(path):
        label = 3
    elif 'night' in str(path):
        label = 4
    elif 'sofa' in str(path):
        label = 25
    elif 'glass' in str(path):
        label = 17
    elif 'cup' in str(path):
        label = 18
    elif 'person' in str(path):
        label = 22
    elif 'range' in str(path):
        label = 35
    elif 'desk' in str(path):
        label = 24
    elif 'bed' in str(path):
        label = 11
    elif 'toilet' in str(path):
        label = 14
    elif 'laptop' in str(path):
        label = 19
    elif 'mantel' in str(path):
        label = 0
    elif 'xbox' in str(path):
        label = 1
    elif 'monitor' in str(path):
        label = 8
    elif 'stairs' in str(path):
        label = 6
    elif 'table' in str(path):
        label = 12
    elif 'car' in str(path):
        label = 36
    elif 'bowl' in str(path):
        label = 29
    elif 'wardrobe' in str(path):
        label = 2
    elif 'tv' in str(path):
        label = 39
    elif 'cone' in str(path):
        label = 20
    elif 'sink' in str(path):
        label = 27
    elif 'airplane' in str(path):
        label = 26

    mesh = trimesh.load_mesh(path, process=False)

    for method in augments:
        if method == 'orient':
            mesh = randomize_mesh_orientation(mesh)
        if method == 'scale':
            mesh = random_scale(mesh)
        if method == 'deformation':
            mesh = mesh_deformation(mesh)

    F = mesh.faces
    V = mesh.vertices

    Fs = mesh.faces.shape[0]
    face_coordinate = V[F.flatten()].reshape(-1, 9)

    face_center = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1)
    vertex_normals = mesh.vertex_normals
    face_normals = mesh.face_normals
    face_curvs = np.vstack([
        (vertex_normals[F[:, 0]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 1]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 2]] * face_normals).sum(axis=1),
    ])

    feats = []
    if 'area' in request:
        feats.append(mesh.area_faces)
    if 'normal' in request:
        feats.append(face_normals.T)
    if 'center' in request:
        feats.append(face_center.T)
    if 'face_angles' in request:
        feats.append(np.sort(mesh.face_angles, axis=1).T)
    if 'curvs' in request:
        feats.append(np.sort(face_curvs, axis=0))

    feats = np.vstack(feats)
    patch_num = Fs // 4 // 4 // 4
    allindex = np.array(list(range(0, Fs)))
    indices = allindex.reshape(-1, patch_num).transpose(1, 0)

    feats_patch = feats[:, indices]
    center_patch = face_center[indices]
    cordinates_patch = face_coordinate[indices]
    faces_patch = mesh.faces[indices]

    feats_patch = feats_patch
    center_patch = center_patch
    cordinates_patch = cordinates_patch
    faces_patch = faces_patch
    feats_patcha = np.concatenate((feats_patch, np.zeros((10, 256 - patch_num, 64), dtype=np.float32)), 1)
    center_patcha = np.concatenate((center_patch, np.zeros((256 - patch_num, 64, 3), dtype=np.float32)), 0)
    cordinates_patcha = np.concatenate((cordinates_patch, np.zeros((256 - patch_num, 64, 9), dtype=np.float32)), 0)
    faces_patcha = np.concatenate((faces_patch, np.zeros((256 - patch_num, 64, 3), dtype=np.int)), 0)

    Fs_patcha = np.array(Fs)

    return feats_patcha, center_patcha, cordinates_patcha, faces_patcha, Fs_patcha, label
'''
def load_mesh_seg(path, normalize=True,augments=[], request=[], seed=None):
    mesh = trimesh.load_mesh(path, process=False)

    base_name = os.path.basename(path)
    name_without_ext = os.path.splitext(base_name)[0]
    #label_path = os.path.join("../dataset/data_obj_parent_directory", name_without_ext.split('_')[0], base_name)
    label_path = Path(str(path).replace('obj', 'json'))
    #print(label_path)
    with open(label_path) as f:
        segment = json.load(f)

    #point_labels = np.array(segment['labels']) #- 1
    #sub_labels = generate_face_labels_nearest(mesh, point_labels)
    sub_labels = np.array(segment['sub_labels'])
    
    for method in augments:
        if method == 'orient':
            mesh = randomize_mesh_orientation(mesh)
        if method == 'scale':
            mesh = random_scale(mesh)
        if method == 'deformation':
            mesh = mesh_deformation(mesh)
    if normalize:
        mesh = mesh_normalize(mesh)
    F = mesh.faces
    V = mesh.vertices
    Fs = mesh.faces.shape[0]
    face_coordinate = V[F.flatten()].reshape(-1, 9)

    face_center = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1)
    vertex_normals = mesh.vertex_normals
    face_normals = mesh.face_normals
    face_curvs = np.vstack([
        (vertex_normals[F[:, 0]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 1]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 2]] * face_normals).sum(axis=1),
    ])

    feats = []
    if 'area' in request:
        feats.append(mesh.area_faces)
    if 'normal' in request:
        feats.append(face_normals.T)
    if 'center' in request:
        feats.append(face_center.T)
    if 'face_angles' in request:
        feats.append(np.sort(mesh.face_angles, axis=1).T)
    if 'curvs' in request:
        feats.append(np.sort(face_curvs, axis=0))

    feats = np.vstack(feats)
    patch_num = Fs // 4 // 4 // 4 #=256
    if patch_num != 256:
        print(f"wrongpath:{path}")
        print(f"Fs:{Fs}")
        return 
    allindex = np.array(list(range(0, Fs)))
    indices = allindex.reshape(-1, patch_num).transpose(1, 0)

    #print(path)
    #print(f"len_label:{len(sub_labels)}")
    #print(f"vertices:{mesh.vertices.shape[0]}")
    #print(f"faces:{Fs}")

    feats_patch = feats[:, indices]
    center_patch = face_center[indices]
    cordinates_patch = face_coordinate[indices]
    faces_patch = mesh.faces[indices]
    label_patch = sub_labels[indices]
    label_patcha = np.concatenate((label_patch, np.zeros((256 - patch_num, 64), dtype=np.float32)), 0)
    label_patcha = np.expand_dims(label_patcha, axis=2)
    label_patcha[label_patcha < 0] = 0  ## 0 is the background class
    feats_patch = feats_patch
    center_patch = center_patch
    cordinates_patch = cordinates_patch
    faces_patch = faces_patch
    feats_patcha = np.concatenate((feats_patch, np.zeros((13, 256 - patch_num, 64), dtype=np.float32)), 1)
    center_patcha = np.concatenate((center_patch, np.zeros((256 - patch_num, 64, 3), dtype=np.float32)), 0)
    cordinates_patcha = np.concatenate((cordinates_patch, np.zeros((256 - patch_num, 64, 9), dtype=np.float32)), 0)
    faces_patcha = np.concatenate((faces_patch, np.zeros((256 - patch_num, 64, 3), dtype=np.float32)), 0)
    feats_patcha = feats_patcha.transpose(1, 2, 0)
    Fs_patcha = np.array(Fs)
    Fs_patcha = Fs_patcha.repeat(256 * 64).reshape(256, 64, 1)

    return faces_patcha, feats_patcha, Fs_patcha, center_patcha, cordinates_patcha, label_patcha

def load_mesh_shape(path, augments=[], request=[], seed=None):

    mesh = trimesh.load_mesh(path, process=False)

    for method in augments:
        if method == 'orient':
            mesh = randomize_mesh_orientation(mesh)
        if method == 'scale':
            mesh = random_scale(mesh)
        if method == 'deformation':
            mesh = mesh_deformation(mesh)

    F = mesh.faces
    V = mesh.vertices

    Fs = mesh.faces.shape[0]
    face_coordinate = V[F.flatten()].reshape(-1, 9)

    face_center = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1)
    vertex_normals = mesh.vertex_normals
    face_normals = mesh.face_normals
    face_curvs = np.vstack([
        (vertex_normals[F[:, 0]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 1]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 2]] * face_normals).sum(axis=1),
    ])

    feats = []
    if 'area' in request:
        feats.append(mesh.area_faces)
    if 'normal' in request:
        feats.append(face_normals.T)
    if 'center' in request:
        feats.append(face_center.T)
    if 'face_angles' in request:
        feats.append(np.sort(mesh.face_angles, axis=1).T)
    if 'curvs' in request:
        feats.append(np.sort(face_curvs, axis=0))

    feats = np.vstack(feats)
    patch_num = Fs // 4 // 4 // 4   #256
    if patch_num != 256:
        print(f"wrongpath:{path}")
        print(f"Fs:{Fs}")
        return 
    #print(f"Fs:{Fs}")
    #print(f"patch_num:{patch_num}")
    allindex = np.array(list(range(0, Fs)))
    '''
    # 计算需要补全的零的数量
    padding_size = (patch_num - (Fs % patch_num)) % patch_num

    # 用零填充数组
    if padding_size > 0:
        allindex = np.pad(allindex, (0, padding_size), 'constant', constant_values=0)
    '''
    indices = allindex.reshape(-1, patch_num).transpose(1, 0)

    feats_patch = feats[:, indices]
    center_patch = face_center[indices]
    cordinates_patch = face_coordinate[indices]
    faces_patch = mesh.faces[indices]

    feats_patch = feats_patch
    center_patch = center_patch
    cordinates_patch = cordinates_patch
    faces_patch = faces_patch

    feats_patcha = np.concatenate((feats_patch, np.zeros((13, 256 - patch_num, feats_patch.shape[2]), dtype=np.float32)), 1)   #10 64
    center_patcha = np.concatenate((center_patch, np.zeros((256 - patch_num, center_patch.shape[1], 3), dtype=np.float32)), 0)
    cordinates_patcha = np.concatenate((cordinates_patch, np.zeros((256 - patch_num, cordinates_patch.shape[1], 9), dtype=np.float32)), 0)
    faces_patcha = np.concatenate((faces_patch, np.zeros((256 - patch_num, faces_patch.shape[1], 3), dtype=np.float32)), 0)

    Fs_patcha = np.array(Fs)

    return feats_patcha, center_patcha, cordinates_patcha, faces_patcha, Fs_patcha

class ClassificationDataset(data.Dataset):
    def __init__(self, dataroot, train=True, augment=None):
        super().__init__()

        self.dataroot = Path(dataroot)
        self.augments = []
        self.mode = 'train' if train else 'test'
        self.feats = ['area', 'face_angles', 'curvs', 'normal', 'center']  #-center

        self.mesh_paths = []
        self.labels = []
        self.browse_dataroot()
        if train and augment:
            self.augments = augment
    
    def browse_dataroot(self):
        self.shape_classes = [x.name for x in self.dataroot.iterdir() if x.is_dir()]
        cnt=0
        maxx=200
        '''
        for obj_path in self.dataroot.iterdir():
            if obj_path.is_file() and obj_path.suffix == '.obj':
                self.mesh_paths.append(obj_path)
                cnt+=1
                if cnt>=maxx:
                    break
                #label = self.shape_classes.index(obj_class.name)
                #print(obj_class)
        '''
        
        for obj_path in (self.dataroot / self.mode).iterdir():
            if obj_path.is_file() and obj_path.suffix == '.obj':
                self.mesh_paths.append(obj_path)
                #self.labels.append(label)
        
        if self.mode == 'train': #
            self.mode = 'val'
            for obj_path in (self.dataroot / self.mode).iterdir():
                if obj_path.is_file() and obj_path.suffix == '.obj':
                    self.mesh_paths.append(obj_path)
            self.mode = 'train'
        
        self.mesh_paths = np.array(self.mesh_paths)
        self.labels = np.array(self.labels)
    
    '''
    def browse_dataroot(self):
        if self.mode == 'train': #
            self.mode = 'val'

        print(f"mode:{self.mode}")
        with open(Path(self.dataroot) / f"base_name_{self.mode}_fold.txt", "r") as file:
            folders = file.readlines()
        
        for folder in folders:
            folder = folder.strip()
            obj_dir = Path(self.dataroot) / f"data_obj_parent_directory/{folder}"
            json_dir = Path(self.dataroot) / f"data_json_parent_directory/{folder}"
            
            if obj_dir.is_dir() and json_dir.is_dir():
                for obj_path in obj_dir.iterdir():
                    if obj_path.suffix == '.obj':
                        obj_name = obj_path.stem
                        seg_path = json_dir / (obj_name + '.json')
                        check_dir = Path("../dataset/Data_maps/" + obj_path.name)
                        if seg_path.is_file() and check_dir.is_file():
                            self.mesh_paths.append("../dataset/Data_maps/" + obj_path.name)
                            #print("../dataset/Data_maps/" + obj_path.name)

        self.mesh_paths = np.array(self.mesh_paths)
        if(self.mode == 'val'):
            self.mode = 'train'
    '''
    def __getitem__(self, idx):

        # label = self.labels[idx]
        label = 0
        if self.mode == 'train':

            feats, center, cordinates, faces, Fs = load_mesh_shape(self.mesh_paths[idx], augments=self.augments,
                                                                    request=self.feats)

            feats2, center2, cordinates2, faces2, Fs2 = load_mesh_shape(self.mesh_paths[idx], augments=self.augments,
                                                                     request=self.feats)
            return feats, center, cordinates, faces, Fs, label, str(self.mesh_paths[idx]), feats2, center2, cordinates2, faces2, Fs2
        else:

            feats, center, cordinates, faces, Fs = load_mesh_shape(self.mesh_paths[idx],
                                                                    augments=self.augments,
                                                                    request=self.feats)
            feats2, center2, cordinates2, faces2, Fs2 = load_mesh_shape(self.mesh_paths[idx],
                                                                    augments=self.augments,
                                                                    request=self.feats)
            return feats, center, cordinates, faces, Fs, label, str(self.mesh_paths[idx]), feats2, center2, cordinates2, faces2, Fs2

    def __len__(self):
        return len(self.mesh_paths)


class SegmentationDataset(data.Dataset):
    def __init__(self, dataroot, train=True, augments=None):
        super().__init__()

        self.dataroot = dataroot

        self.augments = []
        # if train and augments:
        # self.augments = augments
        self.augments = augments
        self.mode = 'train' if train else 'test'
        self.feats = ['area', 'face_angles', 'curvs', 'normal', 'center']   #center

        self.mesh_paths = []
        self.raw_paths = []
        self.seg_paths = []
        self.browse_dataroot()

    # self.set_attrs(total_len=len(self.mesh_paths))
    
    def browse_dataroot(self):
        cnt=0
        maxx=200
        print(f'dataroot:{self.dataroot}')
        for json_path in Path(self.dataroot).iterdir():
            if json_path.suffix == '.json':
                json_name = json_path.stem
                obj_path = json_path.parent / (json_name + '.obj')

                self.mesh_paths.append(str(obj_path))
                self.mesh_paths.append(str(obj_path))
                cnt+=1
                if cnt>=maxx:
                    break

        self.mesh_paths = np.array(self.mesh_paths)
    
    '''
    def browse_dataroot(self):
        if self.mode == 'train':
            self.mode = 'val'
        with open(Path(self.dataroot) / f"base_name_{self.mode}_fold.txt", "r") as file:
            folders = file.readlines()
        
        for folder in folders:
            folder = folder.strip()
            obj_dir = Path(self.dataroot) / f"data_obj_parent_directory/{folder}"
            json_dir = Path(self.dataroot) / f"data_json_parent_directory/{folder}"
            
            if obj_dir.is_dir() and json_dir.is_dir():
                for obj_path in obj_dir.iterdir():
                    if obj_path.suffix == '.obj':
                        obj_name = obj_path.stem
                        seg_path = json_dir / (obj_name + '.json')
                        check_dir = Path("../dataset/Data_maps/" + obj_path.name)
                        if seg_path.is_file() and check_dir.is_file():
                            self.mesh_paths.append("../dataset/Data_maps/" + obj_path.name)
                            #print("../dataset/Data_maps/" + obj_path.name)

        self.mesh_paths = np.array(self.mesh_paths)
        if(self.mode == 'val'):
            self.mode = 'train'
    '''
    def __getitem__(self, idx):

        if self.mode == 'train':

            faces_patcha, feats_patcha, Fs_patcha, center_patcha, cordinates_patcha, label_patcha = load_mesh_seg(
                self.mesh_paths[idx],
                normalize=True,
                augments=self.augments,
                request=self.feats)
            return faces_patcha, feats_patcha, Fs_patcha, center_patcha, cordinates_patcha, label_patcha
        else:
            faces_patcha, feats_patcha, Fs_patcha, center_patcha, cordinates_patcha, label_patcha = load_mesh_seg(
                self.mesh_paths[idx],
                normalize=True,
                request=self.feats)
            print(f'Path:{self.mesh_paths[idx]}')
            return faces_patcha, feats_patcha, Fs_patcha, center_patcha, cordinates_patcha, label_patcha

    
    
    def __len__(self):
        return len(self.mesh_paths)


class ShapeNetDataset(data.Dataset):
    def __init__(self, dataroot, train=True, augment=None):
        super().__init__()

        self.dataroot = Path(dataroot)
        self.augments = []
        self.feats = ['area', 'face_angles', 'curvs', 'normal']
        self.mesh_paths = []

        self.browse_dataroot()
        if train and augment:
            self.augments = augment

    def browse_dataroot(self):
        self.shape_classes = [x.name for x in self.dataroot.iterdir() if x.is_dir()]

        for obj_class in self.dataroot.iterdir():
            if obj_class.is_dir():
                for obj_path in (obj_class).iterdir():
                    if obj_path.is_file():
                        self.mesh_paths.append(obj_path)

        self.mesh_paths = np.array(self.mesh_paths)

    def __getitem__(self, idx):
        label = 0
        feats, center, cordinates, faces, Fs = load_mesh_shape(self.mesh_paths[idx], augments=self.augments,
                                                             request=self.feats)
                                                             
        return   feats, center, cordinates, faces, Fs, label, str(self.mesh_paths[idx])



    def __len__(self):
        return len(self.mesh_paths)
