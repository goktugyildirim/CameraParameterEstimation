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


def LM7DoF(object_points, image_points, rvec0, tvec0, camera_matrix, iterations, log=False):
    num_pairs = len(object_points)

    dist_coeff = np.zeros((4, 1))
    imagePoints0, _ = cv2.projectPoints(object_points, rvec0, tvec0, camera_matrix, dist_coeff, jacobian=False)
    imagePoints0 = imagePoints0.reshape(-1, 2)
    err0 = np.linalg.norm(imagePoints0 - image_points)

    if log:
        print("SE3 Initial Translation:", tvec0, " | SE3 Initial Rotation:", rvec0, " | Focal length: ",
              camera_matrix[0, 0],
              " | Reprojection error: ", np.round(err0, 4))
        print("-------------------------------------------")

    lmbda = 1e-3
    err0 = 0
    for iter in range(iterations):
        object_points = object_points.reshape((num_pairs, 3))
        image_points = image_points.reshape((num_pairs, 2))
        projected_points, J = cv2.projectPoints(object_points, rvec0, tvec0,
                                                camera_matrix, dist_coeff, aspectRatio=1,
                                                jacobian=True)

        projected_points = projected_points.reshape((num_pairs, 2))
        J_t = J[:, 3:6]
        J_r = J[:, :3]
        J_ext = np.concatenate((J_t, J_r), axis=1)
        J_f = J[:, 7:8]
        J_calib = np.concatenate((J_ext, J_f), axis=1)
        residuals = image_points - projected_points
        residuals = residuals.reshape((num_pairs * 2, 1))

        err0 = euclidean_dist(projected_points, image_points)

        J_in = J_calib

        H = J_in.T.dot(J_in)
        H_damped = H + lmbda * np.eye(H.shape[0])
        b = J_in.T.dot(residuals)

        update = np.linalg.solve(H_damped, b)

        tvec0[0] += update[0]
        tvec0[1] += update[1]
        tvec0[2] += update[2]
        rvec0[0] += update[3]
        rvec0[1] += update[4]
        rvec0[2] += update[5]
        camera_matrix[0, 0] += update[6]
        camera_matrix[1, 1] += update[6]

        if log:
            print("Iter: ", iter, " | Translation: ", np.round(tvec0, 3),
                  " | Rotation: ", np.round(rvec0, 3), " F: ", camera_matrix[0, 0],
                  " | Reprojection error: ", np.round(err0, 4))

            print("---------------------------------------------------"
                  "--------------------------------------------------------------------------")

        if err0 < 1e-1:
            break

    return rvec0, tvec0, camera_matrix, err0


def euclidean_dist(projected_points, image_points):
    dist_total = 0
    for i in range(len(projected_points)):
        dist_total += math.sqrt((projected_points[i][0] - image_points[i][0]) ** 2 + (
                projected_points[i][1] - image_points[i][1]) ** 2)
    return dist_total/len(projected_points)

# Define ground truth camera parameters:
width = 1920
height = 1080
fy = 1200
fx = fy
cx = width / 2
cy = height / 2
T_world_cam = np.eye(4)
tx = 0.0
ty = 0.0
tz = 0.4
T_world_cam[0, 3] = tx
T_world_cam[1, 3] = ty
T_world_cam[2, 3] = tz
print("Ground truth values: fx:", fx, "fy:", fy, "tx:", -tx, "ty:", -ty, "tz:", -tz)

count_pairs = 150
outlier_ratio = 0.35
err_var = 0.4
outlier_count = int(count_pairs * outlier_ratio)
inlier_count = count_pairs - outlier_count

image_points_inlier, image_points_outlier = [], []
object_points_inlier, object_points_outlier = [], []

# ADD Inlier points
inlier_indices_gt = np.arange(inlier_count, dtype=np.int32)
for i in range(inlier_count):
    # define rand pixel loc
    x = np.random.randint(0, width)
    y = np.random.randint(0, height)
    # define rand depth
    depth = np.random.randint(1, 10)
    point_cam = np.array([(x - cx) * depth / fx, (y - cy) * depth / fy, depth], dtype=np.float32)
    point_world = T_world_cam.dot(np.append(point_cam, 1))[:3]
    image_points_inlier.append(np.array([x, y], dtype=np.float32))
    object_points_inlier.append(point_world)

# define outlier points
while len(image_points_outlier) < outlier_count:
    # define rand pixel loc
    x = np.random.randint(0, width)
    y = np.random.randint(0, height)
    # define rand depth
    depth = np.random.randint(1, 10)
    point_cam = np.array([(x - cx) * depth / fx, (y - cy) * depth / fy, depth], dtype=np.float32)
    point_world = T_world_cam.dot(np.append(point_cam, 1))[:3]
    # add gaussian noise
    point_world[0] += np.random.normal(0, err_var)
    point_world[1] += np.random.normal(0, err_var)
    b_in_image = 0 <= x < width and 0 <= y < height
    if b_in_image:
        image_points_outlier.append(np.array([x, y], dtype=np.float32))
        object_points_outlier.append(point_world)

# define inlier and outlier indices
inlier_indices_array = np.arange(inlier_count, dtype=np.int32)
outlier_indices_array = np.arange(inlier_count, inlier_count + outlier_count, dtype=np.int32)

object_points_combined = np.array(object_points_inlier + object_points_outlier)
image_points_combined = np.array(image_points_inlier + image_points_outlier)

print("Inlier count: ", len(image_points_inlier))
print("Outlier count: ", len(image_points_outlier))

# Define initial values to bre refined
rvec_init = np.array([0, 0, 0.0], dtype=np.float32)
tvec_init = np.array([0, 0, 0], dtype=np.float32)
fy_init = 1600
fx_init = fy_init
camera_matrix_init = np.array([[fy_init, 0, cx], [0, fy_init, cy], [0, 0, 1]], dtype=np.float32)

t_start = time.time()
rvec, tvec, camera_matrix, err = LM7DoF(object_points_combined, image_points_combined,
                                        rvec_init, tvec_init,
                                        camera_matrix_init, 6, True)
print("Last translation: ", tvec, " | Last rotation: ", rvec, " F: ", camera_matrix[0, 0], " | Reprojection error: ", err)
t_end = time.time()
dur_ms_avg = (t_end - t_start)
print("Time for run: ", dur_ms_avg, " ms")

print("\n\n\n\n\n")

# Run LM - with RANSAC
t_start = time.time()
num_run = 100
num_cam_select = 5
ransac_threshold = 3
num_total_cam = object_points_combined.shape[0]

best_err = 1e10
best_inlier_indices = []
best_outlier_indices = []
best_run_id = 0
best_inlier_count = 0
best_tvec = np.array([0, 0, 0], dtype=np.float32)
best_f = 0

for i in range(num_run):
    print("Run: ", i)
    indices = np.random.choice(num_total_cam, num_cam_select, replace=False)
    c_inlier_in, c_outlier_in = 0, 0
    for index in indices:
        if index in inlier_indices_array:
            c_inlier_in += 1
        else:
            c_outlier_in += 1
    print("Selected inlier ", c_inlier_in, " | Selected outlier ", c_outlier_in)

    object_point_ransac = object_points_combined[indices]
    image_point_ransac = image_points_combined[indices]


    rvec_init = np.array([0, 0, 0.0], dtype=np.float32)
    tvec_init = np.array([0, 0, 0], dtype=np.float32)
    fy_init = 1600
    fx_init = fy_init
    camera_matrix_init = np.array([[fy_init, 0, cx], [0, fy_init, cy], [0, 0, 1]], dtype=np.float32)

    rvec, tvec, camera_matrix, err = LM7DoF(object_point_ransac, image_point_ransac,
                                            rvec_init, tvec_init, camera_matrix_init, 6, False)

    # Evaluate solution
    c_inlier = 0
    inlier_indices_solution, outlier_indices_solution = [], []
    projected_points = cv2.projectPoints(object_points_combined, rvec, tvec, camera_matrix, None)[0]
    for j in range(num_total_cam):
        err_sample = np.linalg.norm(image_points_combined[j] - projected_points[j])
        err_sample = math.sqrt(err_sample)
        if err_sample < ransac_threshold:
            inlier_indices_solution.append(j)
            c_inlier += 1
        else:
            outlier_indices_solution.append(j)

    print("Solution inlier count: ", len(inlier_indices_solution), " | Solution outlier count: ", len(outlier_indices_solution))

    if c_inlier > best_inlier_count:
        best_inlier_count = c_inlier
        best_err = err
        best_tvec = tvec
        best_f = camera_matrix[0, 0]
        best_inlier_indices = inlier_indices_solution
        best_outlier_indices = outlier_indices_solution
        best_run_id = i




    print("------------------------------------------------------------------------------------------------")

print("\n\n\n")

print("Best inlier count: ", best_inlier_count, " | Best outlier count: ", object_points_combined.shape[0] - best_inlier_count)
print("Best run:", best_run_id, " | Best error: ", np.round(best_err, 3), " | Best translation: ", np.round(best_tvec, 3), " | Best F: ", best_f)

print("Outlier intersection ratio: ", len(set(best_outlier_indices).intersection(set(outlier_indices_array)))/len(outlier_indices_array))


t_end = time.time()
dur_ms = (t_end - t_start) * 1000
print("Time for run: ", dur_ms, " ms")





