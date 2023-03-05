import math

import sophus
import numpy as np
import cv2
import time


def convert_pose_to_se3(rvec, tvec):
    # Create rotation matrix from rotation vector
    R = sophus.SO3.exp(rvec).matrix()
    # Create translation vector
    t = np.array([tvec[0], tvec[1], tvec[2]])
    # Create SE3 transformation object
    T = sophus.SE3(R, t)
    return T


def project_point(object_point, rvec, tvec, camera_matrix):
    transform_world_to_camera = np.eye(4)
    mat_rot = np.eye(3)
    cv2.Rodrigues(rvec, mat_rot)
    transform_world_to_camera[:3, :3] = mat_rot
    transform_world_to_camera[:3, 3] = tvec
    X_cam = transform_world_to_camera.dot(np.array([object_point[0], object_point[1], object_point[2], 1]))
    x = X_cam[0] / X_cam[2]
    y = X_cam[1] / X_cam[2]
    u = camera_matrix[0, 0] * x + camera_matrix[0, 2]
    v = camera_matrix[1, 1] * y + camera_matrix[1, 2]
    return u, v


def LM7DoF(object_points, image_points, rvec0, tvec0, camera_matrix, iterations):
    num_pairs = len(object_points)

    dist_coeff = np.zeros((4, 1))
    imagePoints0, _ = cv2.projectPoints(object_points, rvec0, tvec0, camera_matrix, dist_coeff, jacobian=False)
    imagePoints0 = imagePoints0.reshape(-1, 2)
    err0 = np.linalg.norm(imagePoints0 - image_points)

    print("SE3 Initial Translation: \n", tvec0)
    print("SE3 Initial Rotation: \n", rvec0)
    print("Initial focal length: ", camera_matrix[0, 0], camera_matrix[1, 1])
    print("Initial reprojection error: ", err0)
    print("-------------------------------------------")

    lmbda = 1e-3
    total_time_jac_compute = 0
    total_time_solve = 0
    t_start0 = time.time()
    for i in range(iterations):
        object_points = object_points.reshape((num_pairs, 3))
        image_points = image_points.reshape((num_pairs, 2))
        t_start = time.time()
        projected_points, J = cv2.projectPoints(object_points, rvec0, tvec0,
                                                camera_matrix, dist_coeff, aspectRatio=1,
                                                jacobian=True)
        t_end = time.time()
        total_time_jac_compute += np.round((t_end - t_start) * 1000, 3)
        projected_points = projected_points.reshape((num_pairs, 2))
        J_t = J[:, 3:6]
        J_r = J[:, :3]
        J_ext = np.concatenate((J_t, J_r), axis=1)
        J_f = J[:, 7:8]
        J_calib = np.concatenate((J_ext, J_f), axis=1)
        delta = projected_points - image_points
        delta = delta.reshape((num_pairs * 2, 1))

        err0 = np.linalg.norm(image_points - projected_points)

        J_in = J_calib

        t_start = time.time()
        H = J_in.T.dot(J_in)
        H_damped = H + lmbda * np.eye(H.shape[0])
        b = -J_in.T.dot(delta)

        update = np.linalg.solve(H_damped, b)

        t_end = time.time()
        total_time_solve += np.round((t_end - t_start) * 1000, 3)

        tvec0[0] += update[0]
        tvec0[1] += update[1]
        tvec0[2] += update[2]
        rvec0[0] += update[3]
        rvec0[1] += update[4]
        rvec0[2] += update[5]
        camera_matrix[0, 0] += update[6]
        camera_matrix[1, 1] += update[6]

        print("Translation: ", np.round(tvec0, 3), " | Rotation: ",
              np.round(rvec0, 3), " F: ", camera_matrix[0, 0], " | Reprojection error: ", np.round(err0, 4))

        print("---------------------------------------------------"
              "--------------------------------------------------------------------------")

        if err0 < 1e-1:
            break
    t_end0 = time.time()
    dur0_ms = np.round((t_end0 - t_start0) * 1000, 3)

    print("Total time for Jacobian computation: ", np.round(total_time_jac_compute, 3), " ms")
    print("Total time for solving the linear system: ", np.round(total_time_solve, 3), " ms")
    print("Total time for the whole process: ", dur0_ms, " ms")

    return rvec0, tvec0, camera_matrix


# I derive the Jacobian for the PnP problem
def pnp_gauss_newton_exp_map(object_points, image_points, rvec, tvec, camera_matrix, iterations):
    num_pairs = len(object_points)
    se3 = convert_pose_to_se3(rvec, tvec)
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    print("SE3 Initial Translation: \n", se3.translation())
    print("SE3 Initial Rotation: \n", se3.rotationMatrix())

    print("-------------------------------------------")

    for i in range(iterations):
        H = np.zeros((6, 6))
        b = np.zeros((6, 1))
        cost = 0
        for j in range(num_pairs):
            object_point = object_points[j].reshape(3, 1)
            pc = se3 * object_point
            inv_z = 1.0 / pc[2]
            inv_z2 = inv_z ** 2
            proj = np.zeros((2, 1))
            proj[0] = fx * pc[0] / pc[2] + cx
            proj[1] = fy * pc[1] / pc[2] + cy
            e = image_points[j].reshape(2, 1) - proj
            cost += math.sqrt(e[0] ** 2 + e[1] ** 2)

            J = np.zeros((2, 6))
            J[0, 0] = -fx * inv_z
            J[0, 1] = 0
            J[0, 2] = fx * pc[0] * inv_z2
            J[0, 3] = fx * pc[0] * pc[1] * inv_z2
            J[0, 4] = -fx - fx * pc[0] * pc[0] * inv_z2
            J[0, 5] = fx * pc[1] * inv_z

            J[1, 0] = 0
            J[1, 1] = -fy * inv_z
            J[1, 2] = fy * pc[1] * inv_z2
            J[1, 3] = fy + fy * pc[1] * pc[1] * inv_z2
            J[1, 4] = -fy * pc[0] * pc[1] * inv_z2
            J[1, 5] = -fy * pc[0] * inv_z

            H += J.T.dot(J)
            b += -J.T.dot(e)

        print("Iter:", i, " | Exp Cost:", cost / num_pairs)

        # Solve Hx = b
        dx = np.linalg.solve(H, b)
        dx_exp = sophus.SE3.exp(dx)

        # Update pose estimate
        se3 = dx_exp * se3

    print("-------------------------------------------")
    print("SE3 Final Translation: \n", se3.translation())
    print("SE3 Final Rotation: \n", se3.rotationMatrix())

    return se3.translation(), se3.rotationMatrix()


width = 1920
height = 1080
fy = 1200
fx = fy
cx = width / 2
cy = height / 2

image_points = []
object_points = []

T_world_cam = np.eye(4)
tx = 0.15
ty = 0.19
tz = 0.233
T_world_cam[0, 3] = tx
T_world_cam[1, 3] = ty
T_world_cam[2, 3] = tz

print("Ground truth values: fx:", fx, "fy:", fy, "tx:", -tx, "ty:", -ty, "tz:", -tz)

count_pairs = 100
for i in range(count_pairs):
    # define rand pixel loc
    x = np.random.randint(0, width)
    y = np.random.randint(0, height)
    # define rand depth
    depth = np.random.randint(1, 10)
    point_cam = np.array([(x - cx) * depth / fx, (y - cy) * depth / fy, depth], dtype=np.float32)
    point_world = T_world_cam.dot(np.append(point_cam, 1))[:3]
    image_points.append(np.array([x, y], dtype=np.float32))
    object_points.append(point_world)

rvec_init = np.array([0, 0, 0], dtype=np.float32)
tvec_init = np.array([0, 0, 0.0], dtype=np.float32)

fy_init = 1600
fx_init = fy_init
camera_matrix_init = np.array([[fy_init, 0, cx], [0, fy_init, cy], [0, 0, 1]], dtype=np.float32)

rvec, tvec, camera_matrix = LM7DoF(np.array(object_points), np.array(image_points), rvec_init, tvec_init,
                                   camera_matrix_init, 6)
                                   
                                   
