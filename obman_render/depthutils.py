import numpy as np
import cv2


def convert_depth(tmp_depth):
    """
    Prepares for saving by putting infinity depth to 0 and scaling
    valid depth values between 1 and 255
    """
    depth3channels = cv2.imread(tmp_depth, flags=3)
    if depth3channels is None:
        raise ValueError('Could not read image {}'.format(tmp_depth))
    depth = depth3channels[:, :, 0]
    dead_pixels = depth == 1e10
    depth_max = depth[~dead_pixels].max()
    depth_min = depth[~dead_pixels].min()
    scaled_depth = 254 * (depth - depth_min) / (depth_max - depth_min) + 1
    scaled_depth[dead_pixels] = 0
    return scaled_depth, depth_max, depth_min


def read_depth(depth_path, depth_min, depth_max):
    """!!! Untested"""
    depth = cv2.imread(depth_path)
    assert depth.max(
    ) == 255, 'Max value of depth jpg should be 255, not {}'.format(
        depth.max())
    scaled_depth = (depth - 1) / 254 * (depth_max - depth_min) + depth_min
    return scaled_depth

def get_visible(tmp_depth, hand_infos):
    depth3channels = cv2.imread(tmp_depth, flags=3)
    if depth3channels is None:
        raise ValueError('Could not read image {}'.format(tmp_depth))
    depth = depth3channels[:, :, 0]
    dead_pixels = depth == 1e10
    depth_max = depth[~dead_pixels].max()
    depth_min = depth[~dead_pixels].min()
    depth[dead_pixels] = 0

    left_z = hand_infos['lcoords_3d'][:,-1]
    right_z = hand_infos['rcoords_3d'][:,-1]
    left_coord = hand_infos['lcoords_2d']
    right_coord = hand_infos['rcoords_2d']

    lvisible = []
    rvisible = []
    for i in range(left_coord.shape[0]):
        # print('=============get_visible for문==============')
        # print((left_coord.shape[0]))
        # print(left_z[i])
        
        
        x, y = left_coord[i,:]
        x, y = int(x), int(y)
        # print('x:{}, y:{}'.format(x,y))
        # print(depth[x,y])
        # print('====================================')
        # print(i,'joint:', left_z[i], ', image:',depth[x,y])
        # depth[x, y] == depth on image, left_z = joint_depth
        if depth[x,y]+0.015 < left_z[i]:
            lvisible.append(0)
        else:
            lvisible.append(1)
    
    for i in range(right_coord.shape[0]):
        x, y = right_coord[i]
        x, y = int(x), int(y)
        print(i+21,'joint:', left_z[i], ', image:',depth[x,y])
        if depth[x,y]+0.015 < right_z[i]:
            rvisible.append(0)
        else:
            rvisible.append(1)
    
    return lvisible, rvisible














