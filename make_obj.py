import numpy as np
import time
import copy

def make_obj():
    import os
    import random
    import pickle
    import sys

    import bpy
    from sacred import Experiment
    import cv2
    import numpy as np
    from mathutils import Matrix

    root = '.'
    sys.path.insert(0, root)
    mano_path = os.environ.get('MANO_LOCATION', None)
    mano_path = '/home/unist/hdd_ext/hdd_ext/hdd3000/make_db/mano_v1_2'
    smpl_model_path = os.path.join(mano_path, 'models', 'SMPLH_female.pkl')
    mano_right_path = os.path.join(mano_path, 'models', 'MANO_RIGHT.pkl')

    if mano_path is None:
        raise ValueError('Environment variable MANO_LOCATION not defined'
                        'Please follow the README.md instructions')
    sys.path.insert(0, os.path.join(mano_path, 'webuser'))

    from obman_render import (mesh_manip, render, texturing, conditions, imageutils,
                            camutils, coordutils, depthutils)
    from smpl_handpca_wrapper import load_model as smplh_load_model
    from serialization import load_model

    smpl_data_path = 'assets/SURREAL/smpl_data/smpl_data.npz'
    # Load SMPL+H model
    ncomps = 2 * 7  # 2x6 for 2 hands and 6 PCA components
    smplh_model = smplh_load_model(
        smpl_model_path, ncomps=ncomps, flat_hand_mean=False)
    smpl_data = np.load(smpl_data_path)
    hand_pose=np.array([0.07, 1.94, 1.04, 1.85, 1.52, 1.11, 1.29])

    # Path to folder where to render
    results_root = 'results'
    # in ['train', 'test', 'val']
    split = 'train'
    # Number of frames to render
    frame_nb = 1
    # Idx of first frame
    frame_start = 0
    # Min distance to camera
    z_min = 0.5
    # Max distance to camera
    z_max = 0.8
    # Zoom to increase resolution of textures
    texture_zoom = 1
    # combination of [imagenet|lsun|pngs|jpgs|with|4096]
    texture_type = ['bodywithands']
    # Render full bodys and save body annotation
    render_body = False
    high_res_hands = False
    # Combination of [black|white|imagenet|lsun]
    background_datasets = ['imagenet', 'lsun']
    # Paths to background datasets
    lsun_path = '/sequoia/data2/gvarol/datasets/LSUN/data/img'
    imagenet_path = '/sequoia/data3/datasets/imagenet'
    # Lighting ambiant mean
    ambiant_mean = 0.7
    # Lighting ambiant add
    ambiant_add = 0.5
    z_min = 0.5
    # Max distance to camera
    z_max = 0.8
    hand_pose_offset = 0
    hand_pose_var = 2

    smplh_verts, posed_model, meta_info = mesh_manip.randomized_verts(
    smplh_model,
    smpl_data,
    ncomps=ncomps,
    hand_pose=hand_pose,
    z_min=z_min,
    z_max=z_max,
    side='both',
    hand_pose_offset=hand_pose_offset,
    pose_var=hand_pose_var,
    random_shape=True,
    random_pose=True,
    split=split)

    print(smplh_verts)
    print(smplh_verts.shape)
    print(posed_model.shape)
    print(meta_info.keys())
    print(smplh_model.f.shape)

    with open('/home/unist/hdd_ext/hdd_ext/hdd3000/make_db/results/obj/test.obj', 'w') as fp:
        for v in smplh_verts[:]:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in smplh_model.f:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))


def find_vertex():

    ori = 'test.obj'
    arm = 'test.off'

    with open('/home/unist/hdd_ext/hdd_ext/hdd3000/make_db/results/obj/test.obj', 'r') as fp:
        ori = fp.readlines()

    with open('/home/unist/hdd_ext/hdd_ext/hdd3000/make_db/results/obj/test.off', 'r') as fp:
        arm = fp.readlines()

    for idx in range(6890): # 0~6889 is vertices 
        ori[idx] = ori[idx].replace('\n', '').replace('v ', '').split(' ')

    for idx in range(2273): # 0~2272 is vertices
        arm[idx] = arm[idx].replace(' \n', '').split(' ')


    original = [[float(line[0]), float(line[1]), float(line[2])] \
                for line in ori[0:6890]]

    egocentric = [[float(line[0]), float(line[1]), float(line[2])] \
                for line in arm[2:2273]]

    # Find vertex numbers
    indices = []
    for ego_vert in egocentric:
        for idx, ori_vert in enumerate(original):
            if ego_vert[0] == ori_vert[0] and ego_vert[1] == ori_vert[1]:
                indices.append(idx)
                break
            elif ego_vert[2] == ori_vert[2] and ego_vert[1] == ori_vert[1]:
                indices.append(idx)
                break

    indices.sort()
    #print(indices)

    with open("hand_vert.txt", "w") as output:
        for idx in indices:
            output.write(str(idx)+'\n')

def substract_face_num():
    with open('/home/unist/hdd_ext/hdd_ext/hdd3000/make_db/right.obj', 'r') as fp:
        ori = fp.readlines()
    
    for idx in reversed(range(1134, len(ori))): # 0~6889 is vertices
        verts_num = np.array(list(map(int,ori[idx].replace('\n', '').split(' ')[1:])))
        if (verts_num < 1137).any():
            del ori[idx]
        else:
            verts_num = verts_num - 1137
            ori[idx] = 'f {0} {1} {2}\n'.format(verts_num[0],verts_num[1],verts_num[2])
    with open('/home/unist/hdd_ext/hdd_ext/hdd3000/make_db/right2.obj', 'w') as fp:
        fp.writelines(ori)

def fill_hole():
    with open('/home/unist/hdd_ext/hdd_ext/hdd3000/make_db/right.obj', 'r') as fp:
        ori = fp.readlines()

    right = [22,6,7,45,13,12,256,46,47,1130,1129,42,41,1128,1127,43,44,23]
    right = (np.array(right)+1).tolist()
    left = [25,28,47,46,1131,1130,44,45,1132,1133,50,49,259,14,15,48,9,8]
    for i in range(len(right)):
        if i == len(right)//2 or i == len(right)//2 - 1:
            continue
        if i == len(right)-1:
            ori.append('f {0} {1} {2}\n'.format(right[i], right[0], right[len(right)//2]))
            break    
        ori.append('f {0} {1} {2}\n'.format(right[i], right[i+1], right[len(right)//2]))
    with open('/home/unist/hdd_ext/hdd_ext/hdd3000/make_db/right3.obj', 'w') as fp:
        fp.writelines(ori)

def get_arm_vertices():
    
    with open('/home/unist/hdd_ext/hdd_ext/hdd3000/make_db/obman_render-master/hand_vert.txt', 'r') as f:
        lines = f.readlines()

    for idx in range(len(lines)):
        lines[idx] = int(lines[idx].replace('\n', ''))
    #print(len(lines))
    return lines

def get_each_vert():
    
    with open('/home/unist/hdd_ext/hdd_ext/hdd3000/make_db/obman_render-master/hand_vert.txt', 'r') as f:
        lines = f.readlines()

    for idx in range(len(lines)):
        lines[idx] = int(lines[idx].replace('\n', ''))
    left = lines[:1137]
    right = lines[1137:]
    return left, right


def use_only_arms(verts, faces):
    arm_verts_idx = get_arm_vertices()
    start = time.time()
    arm_idx = [0 for x in range(60000)]
    for idx in arm_verts_idx:
        arm_idx[idx] = 1


    # remove faces if faces don't contain verts.
    for idx in reversed(range(len(faces))):
        if arm_idx[faces[idx][0]] == 0 or arm_idx[faces[idx][1]] == 0 \
            or arm_idx[faces[idx][2]] == 0:
            faces = np.delete(faces, idx, axis=0)
            #del faces[idx]
    one = time.time()
    #print('1 :', one-start)
    # remove verts    
    for idx in reversed(range(len(verts))):
        if arm_idx[idx] == 0:
            verts = np.delete(verts, idx, axis=0)
            #print(type(faces))
            #del verts[idx]
            faces = np.where(faces >= idx, faces-1, faces)
            #for i, face in enumerate(faces):
            #    for j, f in enumerate(face):
            #        if faces[i][j] <= idx:
            #            faces[i][j] -= 1
    two = time.time()
    #print('2 :', two-one)

    return verts, faces

def get_left_right(verts, faces):
    left, right = get_each_vert()
    left_idx = [0 for x in range(60000)]
    right_idx = [0 for x in range(60000)]

    #print(type(faces)) # numpy array (13776,3)
    #print(faces.shape)
    for idx in left:
        left_idx[idx] = 1
    for idx in right:
        right_idx[idx] = 1
    
    left_face = copy.deepcopy(faces)
    right_face = copy.deepcopy(faces)
    left_vert = copy.deepcopy(verts)
    right_vert = copy.deepcopy(verts)
    # remove faces if faces don't contain verts.
    for idx in reversed(range(len(left_face))):
        if left_idx[faces[idx][0]] == 0 or left_idx[faces[idx][1]] == 0 \
            or left_idx[faces[idx][2]] == 0:
            left_face = np.delete(left_face, idx, axis=0)
    
    for idx in reversed(range(len(right_face))):
        if right_idx[faces[idx][0]] == 0 or right_idx[faces[idx][1]] == 0 \
            or right_idx[faces[idx][2]] == 0:
            right_face = np.delete(right_face, idx, axis=0)
    
    # remove verts    
    for idx in reversed(range(len(left_vert))):
        if left_idx[idx] == 0:
            left_vert = np.delete(left_vert, idx, axis=0)
            left_face = np.where(left_face >= idx, left_face-1, left_face)

    for idx in reversed(range(len(right_vert))):
        if right_idx[idx] == 0:
            right_vert = np.delete(right_vert, idx, axis=0)
            right_face = np.where(right_face >= idx, right_face-1, right_face)

    # close meshes
    r_tip = [22,6,7,45,13,12,256,46,47,1130,1129,42,41,1128,1127,43,44,23]
    #r_tip = (np.array(r_tip)+1).tolist()
    l_tip = [25,28,47,46,1131,1130,44,45,1132,1133,50,49,259,14,15,48,9,8]
    #l_tip = (np.array(l_tip)+1).tolist()
    for i in range(len(r_tip)):
        if i == len(r_tip)//2 or i == len(r_tip)//2 - 1:
            continue
        if i == len(r_tip)-1:
            right_face = np.append(right_face, np.array([r_tip[i],r_tip[0],r_tip[len(r_tip)//2]]).reshape(1,-1), axis=0)
            break
        right_face = np.append(right_face, np.array([r_tip[i],r_tip[i+1],r_tip[len(r_tip)//2]]).reshape(1,-1), axis=0)

    for i in range(len(l_tip)):
        if i == len(l_tip)//2 or i == len(l_tip)//2 - 1:
            continue
        if i == len(l_tip)-1:
            left_face = np.append(left_face, np.array([l_tip[i],l_tip[0],l_tip[len(l_tip)//2]]).reshape(1,-1), axis=0)
            break
        left_face = np.append(left_face, np.array([l_tip[i],l_tip[i+1],l_tip[len(l_tip)//2]]).reshape(1,-1), axis=0)

    # with open('/home/donguk/make_db/results/obj/left.obj', 'w') as fp:
    #     for v in left_vert:
    #         fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    #     for f in left_face:  # Faces are 1-based, not 0-based in obj files
    #         fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))


    # with open('/home/donguk/make_db/results/obj/right.obj', 'w') as fp:
    #     for v in right_vert:
    #         fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    #     for f in right_face:  # Faces are 1-based, not 0-based in obj files
    #         fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))
    
    # obj_triangles: (triangle_nb, vertex_nb=3, vertex_coords=3)
    # make triangles
    left_tri = []
    for f in left_face:
        left_tri.append([left_vert[f[0]], left_vert[f[1]], left_vert[f[2]]])
    left_tri = np.array(left_tri)

    right_tri = []
    for f in right_face:
        right_tri.append([right_vert[f[0]], right_vert[f[1]], right_vert[f[2]]])
    right_tri = np.array(right_tri)


    return left_tri, right_tri, left_vert, right_vert




if __name__ == '__main__':
    #find_vertex()
    #show_only_arms()
    #use_only_arms()
    # substract_face_num()
    #fill_hole()
    #get_arm_vertices()
    left, right = get_each_vert()
    #print(len(left))
    #print(len(right))


















