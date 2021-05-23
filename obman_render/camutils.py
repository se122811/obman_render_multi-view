import bpy
from mathutils import Matrix, Vector


def set_camera(camera_name='Camera'):
    # set camera properties and initial position
    bpy.ops.object.select_all(action='DESELECT')
    cam_ob = bpy.data.objects[camera_name]
    scene = bpy.context.scene  # bpy.data.scenes['Scene'] #
    # scene = bpy.data.scenes['Scene']  # Starts from default blender scene
    cam_ob.select = True
    #cam_ob.select_set(state=True)
    cam_ob.animation_data_clear()
    cam_ob.data.lens = 60  # import math  cam_ob.data.angle = math.radians(40)
    cam_ob.data.clip_start = 0.1
    cam_ob.data.sensor_width = 32
    scene.camera.location = Vector((0, 0, 0))
    scene.camera.rotation_euler = Vector((0, 0, 0))

def set_camera(camera_name='Camera',loc=(0,0,0),rot=(0,0,0)):
    # set camera properties and initial position
    bpy.ops.object.select_all(action='DESELECT')
    cam_ob = bpy.data.objects[camera_name]
    scene = bpy.context.scene  # bpy.data.scenes['Scene'] #
    # scene = bpy.data.scenes['Scene']  # Starts from default blender scene
    cam_ob.select = True
    #cam_ob.select_set(state=True)
    cam_ob.animation_data_clear()
    cam_ob.data.lens = 60  # import math  cam_ob.data.angle = math.radians(40)
    cam_ob.data.clip_start = 0.1
    cam_ob.data.sensor_width = 32
    #scene.camera.location = Vector((0, 0, 0))
    #scene.camera.rotation_euler = Vector((0, 0, 0))
    scene.camera.location = Vector(loc)
    scene.camera.rotation_euler = Vector(rot)
    cam_ob.location = Vector(loc)
    cam_ob.rotation_euler = Vector(rot)


def check_camera(camera_name='Camera'):
    """
    We want the camera matrix to be identity in world coordinates
    to simplify further computations
    """
    cam_ob = bpy.data.objects[camera_name]
    id_4 = Matrix(((1., 0., 0., 0), (0., 1., 0., 0.), (0., 0., 1., 0.),
                   (0., 0., 0., 1.)))
    err_msg = 'World camera matrix should be identity, got {}'.format(
        cam_ob.matrix_world)
    assert cam_ob.matrix_world == id_4, err_msg


def get_calib_matrix(cam_name='Camera'):
    scene = bpy.context.scene
    cam_obj = bpy.data.objects[cam_name]
    f_mm = cam_obj.data.lens
    res_x_px = scene.render.resolution_x
    res_y_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100

    px_x = scene.render.pixel_aspect_x
    px_y = scene.render.pixel_aspect_y
    px_aspect_ratio = px_x / px_y
    sensor_w_mm = cam_obj.data.sensor_width
    sensor_h_mm = cam_obj.data.sensor_height

    assert cam_obj.data.sensor_fit == 'AUTO', 'sensor fit should be AUTO, got {}'.format(
        cam_obj.data.sensor_fit)

    # Correct sensor size according to blender code
    # https://github.com/dfelinto/blender/blob/master/intern/cycles/blender/blender_camera.cpp
    sensor_size = sensor_w_mm
    horizontal_fit = res_x_px > res_y_px
    fit_x_ratio = res_x_px * px_x
    fit_y_ratio = res_y_px * px_y
    if horizontal_fit:
        sensor_w_mm = sensor_size
        sensor_h_mm = sensor_size * fit_y_ratio / fit_x_ratio
    else:
        sensor_h_mm = sensor_size
        sensor_w_mm = sensor_size * fit_x_ratio / fit_y_ratio

    # Sanity checks
    assert px_aspect_ratio == 1, 'Pixel aspect ratio should be 1, got {}'.format(
        px_aspect_ratio)

    fx_px = f_mm * res_x_px * scale / sensor_w_mm
    fy_px = f_mm * res_y_px * scale / sensor_h_mm
    err_msg = 'x and y pixel focals should be equal, got {} and {}'.format(
        fx_px, fy_px)
    assert fx_px == fy_px, err_msg

    u_0 = res_x_px * scale / 2
    v_0 = res_y_px * scale / 2

    skew = 0  # Rectangular pixels

    calib_mat = Matrix(((fx_px, skew, u_0), (0, fy_px, v_0), (0, 0, 1)))

    return calib_mat


def get_extrinsic(cam_name='Camera'):
    cam_obj = bpy.data.objects[cam_name]

    # Get world2blender rotation from 3x3 top left corner from matrix_world
    R_world2b = Matrix(
        (cam_obj.matrix_world[0][:3], cam_obj.matrix_world[1][:3],
         cam_obj.matrix_world[2][:3]))

    # Blender and computer vision have different rotations
    R_b2cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))

    # Multiply the two to get world to our usual computer vision setting
    R_world2cv = R_b2cv * R_world2b
    assert cam_obj.location == Vector((0, 0, 0))
    extrinsic = Matrix(
        ((R_world2cv[0][:3] + (0, )), (R_world2cv[1][:3] + (0, )),
         (R_world2cv[2][:3] + (0, ))))

    return extrinsic


# def get_extrinsic(cam_name='Camera',loc=(0,0,0)):
#     print('==========get_extrinsic=================')
#     cam_obj = bpy.data.objects[cam_name]

#     # Get world2blender rotation from 3x3 top left corner from matrix_world
#     R_world2b = Matrix(
#         (cam_obj.matrix_world[0][:3], cam_obj.matrix_world[1][:3],
#          cam_obj.matrix_world[2][:3]))
#     print('=========R_world2b===========')
#     print(R_world2b)
#     # Blender and computer vision have different rotations
#     R_b2cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))

#     # Multiply the two to get world to our usual computer vision setting
#     R_world2cv = R_b2cv * R_world2b
#     #assert cam_obj.location == Vector((0, 0, 0))
#     extrinsic = Matrix(
#         ((R_world2cv[0][:3] + (-cam_obj.matrix_world[0][3], )), (R_world2cv[1][:3] + (-cam_obj.matrix_world[1][3], )),
#          (R_world2cv[2][:3] + (-cam_obj.matrix_world[0][3], ))))
#     print("=============cam_obj.matrix_world===================")
#     print(cam_obj.matrix_world)
#     print('=============scene.camera.location==========')
#     scene = bpy.context.scene
#     print(scene.camera.location)
#     print(scene.camera.rotation_euler)
#     return extrinsic

def get_extrinsic(cam_name='Camera',loc=(0,0,0)):
    cam = bpy.data.objects[cam_name]
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam * location

    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv*R_world2bcam
    T_world2cv = R_bcam2cv*T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT