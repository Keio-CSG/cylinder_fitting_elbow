import open3d as o3d
import argparse
import os
import json
import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
from scipy.optimize import minimize

def cylinderFitting(xyz,p,th):

    """
    This is a fitting for a vertical cylinder fitting
    Reference:
    http://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XXXIX-B5/169/2012/isprsarchives-XXXIX-B5-169-2012.pdf

    xyz is a matrix contain at least 5 rows, and each row stores x y z of a cylindrical surface
    p is initial values of the parameter;
    p[0] = Xc, x coordinate of the cylinder centre
    P[1] = Yc, y coordinate of the cylinder centre
    P[2] = alpha, rotation angle (radian) about the x-axis
    P[3] = beta, rotation angle (radian) about the y-axis
    P[4] = r, radius of the cylinder

    th, threshold for the convergence of the least squares

    from https://stackoverflow.com/a/44164662, partially changed

    """   
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]

    fitfunc = lambda p, x, y, z: (- np.cos(p[3])*(p[0] - x) - z*np.cos(p[2])*np.sin(p[3]) - np.sin(p[2])*np.sin(p[3])*(p[1] - y))**2 + (z*np.sin(p[2]) - np.cos(p[2])*(p[1] - y))**2 #fit function
    errfunc = lambda p, x, y, z: np.linalg.norm(np.sqrt(np.abs(fitfunc(p, x, y, z) - p[4]**2))) #error function 

    print("first error", np.linalg.norm(errfunc(p, x, y, z)))

    # print(p)
    result = minimize(errfunc, p, args=(x, y, z), method="COBYLA", options={'maxiter': 10000}, constraints=({'type': 'ineq', 'fun': lambda p: p[4] - 1}))
    # print(result)

    print("final error", np.linalg.norm(errfunc(result.x, x, y, z)))

    return result.x


def gen_intrinsics():
    intr = rs.pyrealsense2.intrinsics()
    intr.width = 640
    intr.height = 360
    intr.ppx = 321.2279968261719
    intr.ppy = 176.7667999267578
    intr.fx = 318.4465637207031
    intr.fy = 318.4465637207031
    intr.model = rs.pyrealsense2.distortion.brown_conrady
    intr.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
    return intr

def get_keypoints(color_frame, depth_frame):
    """
    RGB画像とDepth画像から体の関節点をmediapipeで抽出する

    Returns: (landmarks, landmarks_world)
    landmarks: 体の関節点のリスト(x,yは画像のピクセル、zはmm単位)
    landmarks_world: 体の関節点の世界座標のリスト(z座標はでたらめ)
    """
    mp_pose = mp.solutions.pose
    landmarks = []
    landmarks_world = []
    with mp_pose.Pose(
        min_detection_confidence=0.5,
    ) as pose:
        results = pose.process(color_frame)
        if results.pose_landmarks:
            image_width = color_frame.shape[1]
            image_height = color_frame.shape[0]
            for landmark in results.pose_landmarks.landmark:
                landmarks.append((
                    int(landmark.x * image_width),
                    int(landmark.y * image_height),
                ))
                if landmark.x > 0 and landmark.y > 0 and landmark.x < 1 and landmark.y < 1:
                    landmarks_world.append(rs.rs2_deproject_pixel_to_point(
                        gen_intrinsics(),
                        [landmark.x * image_width, landmark.y * image_height],
                        depth_frame[landmarks[-1][1], landmarks[-1][0]]
                    ))
                else:
                    landmarks_world.append(rs.rs2_deproject_pixel_to_point(
                        gen_intrinsics(),
                        [landmark.x * image_width, landmark.y * image_height],
                        0
                    ))
            for i in range(len(landmarks)):
                cv2.circle(color_frame, landmarks[i], 2, (0, 0, 255), -1)
                # if i in [12, 14, 16]:
                #     print(i, landmarks[i], depth_frame[landmarks[i][1], landmarks[i][0]])

    return landmarks, landmarks_world

def fit_cylinder_to_bone(pcd, landmark_a, landmark_b, color=None):
    geometries = []
    corners = None
    width = 100 # 腕の幅の半分の px
    if landmark_a[0] == landmark_b[0]:
        corners = np.array([
            [landmark_a[0] + width, landmark_a[1],0],
            [landmark_a[0] - width, landmark_a[1],0],
            [landmark_b[0] - width, landmark_b[1],0],
            [landmark_b[0] + width, landmark_b[1],0],
        ])
    else:
        theta = np.arctan((landmark_b[1] - landmark_a[1]) / (landmark_b[0] - landmark_a[0]))
        dx = np.sin(theta) * width
        dy = np.cos(theta) * width
        corners = np.array([
            [landmark_a[0] - dx, landmark_a[1] + dy,0],
            [landmark_a[0] + dx, landmark_a[1] - dy,0],
            [landmark_b[0] + dx, landmark_b[1] - dy,0],
            [landmark_b[0] - dx, landmark_b[1] + dy,0],
        ])
    
    # 軸に平行な長方形で切り出す場合
    # x_min = min(landmark_a[0], landmark_b[0]) - 50
    # x_max = max(landmark_a[0], landmark_b[0]) + 50
    # y_min = min(landmark_a[1], landmark_b[1]) - 50
    # y_max = max(landmark_a[1], landmark_b[1]) + 50
    # # 腕の部分だけの点群を抽出
    # corners = np.array([
    #     [x_min, y_min, 0],
    #     [x_min, y_max, 0],
    #     [x_max, y_max, 0],
    #     [x_max, y_min, 0],
    # ], dtype=np.float64)
    vol = o3d.visualization.SelectionPolygonVolume()
    vol.orthogonal_axis = "Z"
    vol.axis_max = max(landmark_a[2], landmark_b[2]) + 50
    vol.axis_min = min(landmark_a[2], landmark_b[2]) - 50
    vol.bounding_polygon = o3d.utility.Vector3dVector(corners)
    cropped_pcd = vol.crop_point_cloud(pcd)
    if color is not None:
        cropped_pcd.paint_uniform_color(color)
        geometries.append(cropped_pcd)

    # 円筒を近似
    point_a = (landmark_a[0], landmark_a[1], landmark_a[2])
    point_b = (landmark_b[0], landmark_b[1], landmark_b[2])
    init_radius = 40 # mm
    point_a = (point_a[0], point_a[1], point_a[2] + init_radius)
    point_b = (point_b[0], point_b[1], point_b[2] + init_radius)
    xz_a = (point_a[2] - point_b[2]) / (point_a[0] - point_b[0])
    yz_a = (point_a[2] - point_b[2]) / (point_a[1] - point_b[1])
    xz_b = point_a[0] - point_a[2] / xz_a
    yz_b = point_a[1] - point_a[2] / yz_a
    angle_xz = np.arctan(xz_a)
    angle_yz = np.arctan(-yz_a)
    points = np.asarray(cropped_pcd.points)

    fitted = cylinderFitting(
        points,
        [xz_b, yz_b, angle_yz, angle_xz, init_radius],
        0
    )
    # print(fitted)

    transform = np.array([
        [1,0,0,-fitted[0]],
        [0,1,0,-fitted[1]],
        [0,0,1,0],
        [0,0,0,1]
    ])
    transform = np.dot(np.array([
        [np.cos(-fitted[3]), 0, -np.sin(-fitted[3]), 0],
        [0, 1, 0, 0],
        [np.sin(-fitted[3]), 0, np.cos(-fitted[3]), 0],
        [0, 0, 0, 1]
    ]), transform)
    transform = np.dot(np.array([
        [1, 0, 0, 0],
        [0, np.cos(-fitted[2]), np.sin(-fitted[2]), 0],
        [0, -np.sin(-fitted[2]), np.cos(-fitted[2]), 0],
        [0, 0, 0, 1]
    ]), transform)
    transform = np.dot(np.array([
        [1, 0, 0, fitted[0]],
        [0, 1, 0, fitted[1]],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]), transform)

    origin = np.array([
        fitted[0], fitted[1], 0
    ])
    distance_to_a = np.linalg.norm(origin - np.array(list(point_a)))
    distance_to_b = np.linalg.norm(origin - np.array(list(point_b)))
    cylinder_max = max(distance_to_a, distance_to_b)
    cylinder_min = min(distance_to_a, distance_to_b)
    # print(cylinder_max, cylinder_min)

    points = [
        [fitted[0], fitted[1], cylinder_min - 500],
        [fitted[0], fitted[1], cylinder_max + 500],
    ]
    lines = [[0, 1]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.transform(transform)
    geometries.append(line_set)

    top = np.dot(
        transform, 
        np.array([[fitted[0], fitted[1], 5000, 1]]).T
    )

    points = [
        [fitted[0], fitted[1], 0],
        [top[0], top[1], top[2]]
    ]
    direction_vec = np.array([
        points[1][0] - points[0][0],
        points[1][1] - points[0][1],
        points[1][2] - points[0][2]
    ]).reshape(3,)
    # lines = [[0, 1]]
    # colors = [[1, 0, 0] for i in range(len(lines))]
    # line_set = o3d.geometry.LineSet()
    # line_set.points = o3d.utility.Vector3dVector(points)
    # line_set.lines = o3d.utility.Vector2iVector(lines)
    # line_set.colors = o3d.utility.Vector3dVector(colors)
    # geometries.append(line_set)

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=fitted[4],
        height=cylinder_max - cylinder_min,
        resolution=10
    )
    cylinder.translate([fitted[0], fitted[1], (cylinder_max + cylinder_min)/2])
    cylinder.transform(transform)

    cylinder = o3d.geometry.LineSet.create_from_triangle_mesh(cylinder)
    cylinder.paint_uniform_color([0, 0, 1] if color is None else color)
    geometries.append(cylinder)

    cropped_pcd.translate([0,0,-10])
    return geometries, direction_vec

def calc_cylinder(color_frame, depth_frame, frame_id):

    landmarks, landmarks_world = get_keypoints(color_frame, depth_frame)

    # カラー情報も付けて人を切り出す
    points = []
    colors = []
    max_z_th = max(landmarks_world[12:17:2], key=lambda x: x[2])[2] + 100
    for x in range(depth_frame.shape[1]):
        for y in range(depth_frame.shape[0]):
            if depth_frame[y, x] > 0 and depth_frame[y, x] < max_z_th:
                points.append(rs.rs2_deproject_pixel_to_point(
                    gen_intrinsics(),
                    [x,y], depth_frame[y,x]
                ))
                colors.append(color_frame[y, x][::-1] / 255)
    # Create a point cloud from the frame.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pcd]

    s_e_geom, s_e_direction = fit_cylinder_to_bone(pcd, landmarks_world[12], landmarks_world[14], color=[0, 1, 0])
    geometries.extend(s_e_geom)

    w_e_geom, w_e_direction = fit_cylinder_to_bone(pcd, landmarks_world[16], landmarks_world[14], color=[1, 1, 0])
    geometries.extend(w_e_geom)

    if landmarks_world[12][2] < landmarks_world[14][2]:
        s_e_direction = -s_e_direction
    if landmarks_world[16][2] < landmarks_world[14][2]:
        w_e_direction = -w_e_direction

    # print("s_e_direction", s_e_direction)
    # print("w_e_direction", w_e_direction)

    length_vec_upperarm = np.linalg.norm(s_e_direction)
    length_vec_forearm = np.linalg.norm(w_e_direction)
    inner_product = np.inner(s_e_direction, w_e_direction)
    angle_rad = np.arccos(
        inner_product / (length_vec_upperarm * length_vec_forearm))
    angle_deg = np.rad2deg(angle_rad)

    print("angle:", angle_deg)

    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json", help="configuration file path")
    parser.add_argument("--frame", help="frame id", type=int, default=2)
    args = parser.parse_args()
    json_path = args.json
    frame_id = args.frame
    dir = os.path.split(os.path.abspath(json_path))[0]
    with open(json_path) as f:
        config = json.load(f)
        print(config)

    color_path = os.path.join(dir, config["color_file"])
    depth_path = os.path.join(dir, config["depth_file"])
    frequency = config["frequency"]

    depth_frames = np.load(depth_path)
    depth_frames = depth_frames[depth_frames.files[0]]

    # Load the color and depth frames.
    video = cv2.VideoCapture(color_path)
    color_frames = np.empty(depth_frames.shape + (3,), dtype=np.uint8)
    for i in range(depth_frames.shape[0]):
        ret, color_frames[i,:,:,:] = video.read()

    calc_cylinder(color_frames[frame_id], depth_frames[frame_id], frame_id)
    