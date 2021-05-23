import os
import random
import pickle
import sys

import bpy
from sacred import Experiment
import cv2
import numpy as np
from mathutils import Matrix
from make_obj import use_only_arms, get_arm_vertices


root = '.'
sys.path.insert(0, root)
mano_path = os.environ.get('MANO_LOCATION', None)
mano_path = '/home/unist/hdd_ext/hdd_ext/hdd3000/make_db/mano_v1_2'
if mano_path is None:
    raise ValueError('Environment variable MANO_LOCATION not defined'
                     'Please follow the README.md instructions')
sys.path.insert(0, os.path.join(mano_path, 'webuser'))

from obman_render import (mesh_manip, render, texturing, conditions, imageutils,
                         camutils, coordutils, depthutils)
from smpl_handpca_wrapper import load_model as smplh_load_model
from serialization import load_model

ex = Experiment('generate_dataset')

@ex.config
def exp_config():
    # Path to folder where to render
    results_root = 'results'
    # in ['train', 'test', 'val']
    split = 'train'
    # Number of frames to render
    frame_nb = 1
    # Idx of first frame
    frame_start = 0
    # Min distance to camera
    z_min = 1
    # Max distance to camera
    z_max = 2
    
    # # Min distance to camera
    # z_min = 0.5
    # # Max distance to camera
    # z_max = 0.8
    
    
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

    # hand params
    # pca_comps = 6
    pca_comps = 45
    
    # Pose params are uniform in [-hand_pose_var, hand_pose_var]
    hand_pose_var = 2
    # Path to fit folder
    smpl_data_path = 'assets/SURREAL/smpl_data/smpl_data.npz'
    mano_path = mano_path
    smpl_model_path = os.path.join(mano_path, 'models', 'SMPLH_female.pkl')
    mano_right_path = os.path.join(mano_path, 'models', 'MANO_RIGHT.pkl')



@ex.automain
def run(_config, results_root, split, frame_nb, frame_start, z_min, z_max,
        texture_zoom, texture_type, render_body, high_res_hands,
        background_datasets, lsun_path, imagenet_path, ambiant_mean, ambiant_add,
        hand_pose_var, pca_comps, smpl_data_path, smpl_model_path,
        mano_right_path):
    print(_config)
    #scene = bpy.data.scenes['Scene']
    scene = bpy.data.scenes['Scene']
    
    # Clear default scene cube
    bpy.ops.object.delete()

    # Set results folders
    folder_meta = os.path.join(results_root, 'meta')
    folder_rgb = os.path.join(results_root, 'rgb')
    #folder_rgb = '/home/unist/hdd_ext/hdd_ext/hdd3000/make_db/obman_render-master/out/rgb/'
    folder_segm = os.path.join(results_root, 'segm')
    folder_temp_segm = os.path.join(results_root, 'tmp_segm')
    folder_depth = os.path.join(results_root, 'depth')
    folders = [
        folder_meta, folder_rgb, folder_segm, folder_temp_segm, folder_depth
    ]

    # Create results directories
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    # Load smpl2mano correspondences
    right_smpl2mano = np.load('assets/models/smpl2righthand_verts.npy')
    left_smpl2mano = np.load('assets/models/smpl2lefthand_verts.npy')

    # Load SMPL+H model
    ncomps = 2 * pca_comps  # 2x6 for 2 hands and 6 PCA components
    smplh_model = smplh_load_model(
        smpl_model_path, ncomps=ncomps, flat_hand_mean=False)
    
    #@@@camera camutils
    cam_name = {'Camera1': [(0,0,0),(0,0,0)],
                'Camera2': [(0,0,-3.5),(3.14,0,0)],
                'Camera3':[(0,-1.5,-2),(3.14/2,0,0)],
                'Camera4':[(0,1.4,-2),(-3.14/2,0,0)],
                'Camera5':[(-1.5,0,-2),(0,-3.14/2,0)],
                'Camera6':[(1.5,0,-2),(0,3.14/2,0)]}
    
    #cam_name= {'Camera':[(0,0,0),(0,0,0)]}
    
    #camutils.set_camera()
    #@@@camera@@@
    for name,value in cam_name.items():
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        # print('name: {}'.format(name))
        camutils.set_camera(camera_name=name,loc=value[0],rot=value[1])
    
    backgrounds = imageutils.get_image_paths(
        background_datasets, split=split, lsun_path=lsun_path,
        imagenet_path=imagenet_path)
    print('Got {} backgrounds'.format(len(backgrounds)))

    # Get full body textures
    body_textures = imageutils.get_bodytexture_paths(
        texture_type, split=split, lsun_path=lsun_path,
        imagenet_path=imagenet_path)
    print('Got {} body textures'.format(len(body_textures)))

    # Get high resolution hand textures
    if high_res_hands:
        hand_textures = imageutils.get_hrhand_paths(texture_type, split=split)
        print('Got {} high resolution hand textures'.format(
            len(hand_textures)))
    print('Finished loading textures')

    # Load smpl info
    smpl_data = np.load(smpl_data_path)

    smplh_verts, faces = smplh_model.r, smplh_model.f
    
    # print(len(smplh_verts)) -> 6890
    # print(smplh_verts[0]) -> [ 0.04248498  0.47732165  0.08627268]
    # print(len(faces)) -> 13776
    # print(faces[0]) -> [1 2 0] = start with 0
    smplh_obj = mesh_manip.load_smpl()
    # Smooth the edges of the body model
    bpy.ops.object.shade_smooth()

    # Set camera rendering params
    # scene.render.resolution_x = 256
    # scene.render.resolution_y = 256

    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100

    # Get camera info
    #@@@camera@@@
    for name,value in cam_name.items():
        cam_calib = np.array(camutils.get_calib_matrix(cam_name=name))
        cam_extr = np.array(camutils.get_extrinsic(cam_name=name,loc=value[0],rot=value[1]))
    # cam_calib = np.array(camutils.get_calib_matrix())
    # cam_extr = np.array(camutils.get_extrinsic())

    scs, materials, sh_path = texturing.initialize_texture(
        smplh_obj, texture_zoom=texture_zoom, tmp_suffix='tmp')

    sides = ['right', 'left']

    # Create object material if none is present
    print('Starting loop !')
    np.random.seed(1)
    random.seed(1)
    for i in range(1):#frame_nb):
        frame_idx = i + frame_start

        tmp_files = []  # Keep track of temporary files to delete at the end

        # Sample random hand poses
        side = random.choice(sides)
        hand_pose = None
        hand_pose_offset = 3


        '''If you want to keep same pose, use this code'''
        hand_pose_offset = 0
        #hand_pose=np.array([0.07, 0.94, 1.04, -0.93, 1.75, 1.65, 1.29])
        hand_pose = np.random.randn(90)

        
        # add
        # setattr(smplh_model, 'f', faces)
        # setattr(smplh_model, 'r', smplh_verts)
        # smplh_model = smplh_model._replace(f=faces)
        # smplh_model = smplh_model._replace(r=smplh_verts)
        
        

        # change only verts, face is same
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
        
        # print(len(smplh_verts), len(faces))
        # if i == 0:
        #     smplh_verts, faces = use_only_arms(smplh_verts, faces)
        # else:
        #     smplh_verts, _ = use_only_arms(smplh_verts, faces)
        # print(len(smplh_verts), len(faces))
        # smplh_model.f = faces
        # with open('/home/donguk/make_db/a.obj', 'w') as fp:
        #     for v in smplh_verts[:]:
        #         fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        #     for f in smplh_model.f:  # Faces are 1-based, not 0-based in obj files
        #         fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))

        verts = get_arm_vertices()
        mesh_manip.alter_mesh(smplh_obj, smplh_verts.tolist(), verts)

        print('-------------------posed_model 출력----------------------------')
        print(posed_model.J_transformed.r)
        
        hand_infos = []
        for name,value in cam_name.items():
            print('===================name불러오는 순서 [hand_body_info]=================')
            print('name:{}'.format(name))
            cam_calib = np.array(camutils.get_calib_matrix(cam_name=name))
            cam_extr = np.array(camutils.get_extrinsic(cam_name=name,loc=value[0],rot=value[1]))
        
            hand_info = coordutils.get_hand_body_info(
                posed_model,
                render_body=render_body,
                side='both',
                cam_extr=cam_extr,
                cam_calib=cam_calib,
                right_smpl2mano=right_smpl2mano,
                left_smpl2mano=left_smpl2mano)
            #hand_infos = {**hand_info, **meta_info}
            hand_infos.append({**hand_info, **meta_info})
            print('==============hand_infos 출력====================')
            print(hand_infos)

        frame_prefix = '{:08}'.format(frame_idx)
        
        #@@@camera@@@
        # for name,value in cam_name.items():
        #     camutils.set_camera(camera_name=name,loc=value[0],rot=value[1])
        #camutils.set_camera()
        #camera_name = 'Camera'
        #bpy.ops.object.camera_add()
        #bpy.ops.view3d.object_as_camera()
        # Randomly pick background
        bg_path = random.choice(backgrounds)
        depth_path = os.path.join(folder_depth, frame_prefix)
        tmp_segm_path = render.set_cycle_nodes(
            scene, bg_path, segm_path=folder_temp_segm, depth_path=depth_path)
        print('-------------seg path 출력 ------------------')
        print('folder_temp_segm: {}'.format(folder_temp_segm))
        print('tmp_segm_path: {}'.format(tmp_segm_path))
        tmp_files.append(tmp_segm_path)
        tmp_depth = depth_path + '{:04d}.exr'.format(1)
        tmp_files.append(tmp_depth)
        # Randomly pick clothing texture
        tex_path = random.choice(body_textures)

        # Replace high res hands
        if high_res_hands:
            old_state = random.getstate()
            old_np_state = np.random.get_state()
            hand_path = random.choice(hand_textures)
            tex_path = texturing.get_overlaped(tex_path, hand_path)
            tmp_files.append(tex_path)
            # Restore previous seed state to not interfere with randomness
            random.setstate(old_state)
            np.random.set_state(old_np_state)

        sh_coeffs = texturing.get_sh_coeffs(
            ambiant_mean=ambiant_mean, ambiant_max_add=ambiant_add)
        texturing.set_sh_coeffs(scs, sh_coeffs)

        # Update body+hands image
        tex_img = bpy.data.images.load(tex_path)
        for part, material in materials.items():
            material.node_tree.nodes['Image Texture'].image = tex_img

        # Render
        img_path = os.path.join(folder_rgb, '{}.jpg'.format(frame_prefix))
        scene.render.filepath = img_path
        scene.render.image_settings.file_format = 'JPEG'
        
        for i,cam in enumerate([obj for obj in bpy.data.objects if obj.type == 'CAMERA']):
            if cam.name=='Camera':
                continue
            print("==========for문에서의 카메라 불러오는 순서==============")
            print('cam.name:{}'.format(cam.name))
            bpy.context.scene.camera = cam
            bpy.context.scene.render.filepath = os.path.join('/home/unist/hdd_ext/hdd_ext/hdd3000/make_db/obman_render-master/out/rgb', cam.name)
            bpy.ops.render.render(write_still=True)
            bpy.context.scene.render.filepath = '/home/unist/hdd_ext/hdd_ext/hdd3000/make_db/obman_render-master/out/rgb'
            
            #bpy.ops.render.render(write_still=True)
            
            print('========================img_path출력=======================')
            print(img_path)
            #@@@camera@@@
            for name,value in cam_name.items():
                camutils.check_camera(camera_name=name)
            #camutils.check_camera(camera_name=camera_name)
            segm_img = cv2. imread(tmp_segm_path)[:, :, 0]

            if render_body:
                keep_render = True
            else:
                keep_render = conditions.segm_condition(
                    segm_img, side=side, use_grasps=False)
            print('keep_render {}'.format(keep_render))
            keep_render = True
            depth, depth_min, depth_max = depthutils.convert_depth(tmp_depth)

            print('depth_shape : ', depth.shape)
            print(depth[250,250])
            print('depth_min : ', depth_min)
            print('depth_max : ', depth_max)

            hand_infos[i]['depth_min'] = depth_min
            hand_infos[i]['depth_max'] = depth_max
            hand_infos[i]['bg_path'] = bg_path
            hand_infos[i]['sh_coeffs'] = sh_coeffs
            hand_infos[i]['body_tex'] = tex_path

            #Clean residual files
            if keep_render:
                #Write depth image
                # final_depth_path = os.path.join(folder_depth,
                #                                 '{}.png'.format(frame_prefix))
                final_depth_path = os.path.join(folder_depth,
                                                '{}.png'.format(cam.name))
                cv2.imwrite(final_depth_path, depth)

                # Save meta
                # meta_pkl_path = os.path.join(folder_meta,
                #                             '{}.pkl'.format(frame_prefix))
                meta_pkl_path = os.path.join(folder_meta,
                                            '{}.pkl'.format(cam.name))
                with open(meta_pkl_path, 'wb') as meta_f:
                    pickle.dump(hand_infos[i], meta_f)

                # Write segmentation path
                # segm_save_path = os.path.join(folder_segm,
                #                             '{}.png'.format(frame_prefix))
                segm_save_path = os.path.join(folder_segm,
                                            '{}.png'.format(cam.name))
                cv2.imwrite(segm_save_path, segm_img)
                ex.log_scalar('generated.idx', frame_idx)
            else:
                os.remove(img_path)
            for filepath in tmp_files:
                os.remove(filepath)
    print('DONE')
