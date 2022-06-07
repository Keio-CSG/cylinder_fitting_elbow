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

    # est_p , success = leastsq(errfunc, p, args=(x, y, z), maxfev=10000)
    print(p)
    result = minimize(errfunc, p, args=(x, y, z), method="COBYLA", options={'maxiter': 10000}, constraints=({'type': 'ineq', 'fun': lambda p: p[4] - 1}))
    print(result)

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

def visualize_frame_3d(frame, frame_id):
    """
    Visualize a frame in 3D.
    """
    print(frame.shape)
    print(frame)
    points = []
    for x in range(frame.shape[1]):
        for y in range(frame.shape[0]):
            if frame[y, x] > 0 and frame[y, x] < 1500:
                points.append(rs.rs2_deproject_pixel_to_point(
                    gen_intrinsics(),
                    [x,y], frame[y,x]
                ))
    # Create a point cloud from the frame.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # Visualize the point cloud.
    o3d.visualization.draw_geometries([pcd])

def calc_cylinder(color_frame, depth_frame, frame_id):
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
                        1000
                    ))
            for i in range(len(landmarks)):
                cv2.circle(color_frame, landmarks[i], 2, (0, 0, 255), -1)
                if i in [12, 14, 16]:
                    print(i, landmarks[i], depth_frame[landmarks[i][1], landmarks[i][0]])

    points = []
    colors = []
    for x in range(depth_frame.shape[1]):
        for y in range(depth_frame.shape[0]):
            if depth_frame[y, x] > 0 and depth_frame[y, x] < 1500:
                points.append(rs.rs2_deproject_pixel_to_point(
                    gen_intrinsics(),
                    [x,y], depth_frame[y,x]
                ))
                colors.append(color_frame[y, x][::-1] / 255)
            if x == 256 and y == 192:
                print("elbow", points[-1])
            if x == 340 and y == 156:
                print("shoulder", points[-1])
            if x == 168 and y == 171:
                print("wrist", points[-1])
    # Create a point cloud from the frame.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pcd]

    #############################################################################################肩-肘

    # 腕の部分だけの点群を抽出
    corners = np.array([
        [-300, -100, 0],
        [-300, 100, 0],
        [20, 100, 0],
        [20, -100, 0],
    ], dtype=np.float64)
    vol = o3d.visualization.SelectionPolygonVolume()
    vol.orthogonal_axis = "Z"
    vol.axis_max = 1250
    vol.axis_min = 1050
    vol.bounding_polygon = o3d.utility.Vector3dVector(corners)
    cropped_pcd = vol.crop_point_cloud(pcd)
    # geometries.append(cropped_pcd)

    # 円筒を近似
    shoulder_point = (landmarks_world[12][0], landmarks_world[12][1], landmarks_world[12][2])
    elbow_point = (landmarks_world[14][0], landmarks_world[14][1], landmarks_world[14][2])
    init_radius = 40 # mm
    shoulder_point = (shoulder_point[0], shoulder_point[1], shoulder_point[2] + init_radius)
    elbow_point = (elbow_point[0], elbow_point[1], elbow_point[2] + init_radius)
    xz_a = (shoulder_point[2] - elbow_point[2]) / (shoulder_point[0] - elbow_point[0])
    yz_a = (shoulder_point[2] - elbow_point[2]) / (shoulder_point[1] - elbow_point[1])
    xz_b = shoulder_point[0] - shoulder_point[2] / xz_a
    yz_b = shoulder_point[1] - shoulder_point[2] / yz_a
    angle_xz = np.arctan(xz_a)
    angle_yz = np.arctan(-yz_a)
    points = np.asarray(cropped_pcd.points)

    fitted = cylinderFitting(
        points,
        [xz_b, yz_b, angle_yz, angle_xz, init_radius],
        0
    )
    # fitted = [xz_b, yz_b, angle_yz, angle_xz, init_radius]
    print(fitted)

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

    top = np.dot(
        transform, 
        np.array([[fitted[0], fitted[1], 5000, 1]]).T
    )

    # points = [[fitted[0], fitted[1], 0], [fitted[0] + 1/np.tan(fitted[3]) * 1500, fitted[1] - 1/np.tan(fitted[2]) * 1500, 1500]]
    points = [
        [fitted[0], fitted[1], 0],
        [top[0], top[1], top[2]]
    ]
    lines = [[0, 1]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(line_set)

    # line = np.linspace(0, 1000, 11, dtype=np.float64)
    # points = np.meshgrid(line,line,line,indexing='ij')

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=fitted[4],
        height=10000,
        resolution=10
    )
    cylinder.translate([fitted[0], fitted[1], 0])
    cylinder.transform(transform)

    cylinder = o3d.geometry.LineSet.create_from_triangle_mesh(cylinder)
    cylinder.paint_uniform_color([0, 0, 1])
    geometries.append(cylinder)


    #############################################################################################肘-手首
    # 手首から肘の部分だけの点群を抽出
    corners = np.array([
        [-470, -50, 0],
        [-470, 100, 0],
        [-200, 100, 0],
        [-200, -50, 0],
    ], dtype=np.float64)
    vol = o3d.visualization.SelectionPolygonVolume()
    vol.orthogonal_axis = "Z"
    vol.axis_max = 1100
    vol.axis_min = 950
    vol.bounding_polygon = o3d.utility.Vector3dVector(corners)
    cropped_pcd = vol.crop_point_cloud(pcd)
    # geometries.append(cropped_pcd)

    # 円筒を近似
    wrist_point = (landmarks_world[16][0], landmarks_world[16][1], landmarks_world[16][2])
    elbow_point = (landmarks_world[14][0], landmarks_world[14][1], landmarks_world[14][2])
    init_radius = 40 # mm
    wrist_point = (wrist_point[0], wrist_point[1], wrist_point[2] + init_radius)
    elbow_point = (elbow_point[0], elbow_point[1], elbow_point[2] + init_radius)
    xz_a = (wrist_point[2] - elbow_point[2]) / (wrist_point[0] - elbow_point[0])
    yz_a = (wrist_point[2] - elbow_point[2]) / (wrist_point[1] - elbow_point[1])
    xz_b = wrist_point[0] - wrist_point[2] / xz_a
    yz_b = wrist_point[1] - wrist_point[2] / yz_a
    angle_xz = np.arctan(xz_a)
    angle_yz = np.arctan(-yz_a)
    points = np.asarray(cropped_pcd.points)

    fitted = cylinderFitting(
        points,
        [xz_b, yz_b, angle_yz, angle_xz, init_radius],
        0
    )
    # fitted = [xz_b, yz_b, angle_yz, angle_xz, init_radius]
    print(fitted)

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

    top = np.dot(
        transform, 
        np.array([[fitted[0], fitted[1], 5000, 1]]).T
    )

    # points = [[fitted[0], fitted[1], 0], [fitted[0] + 1/np.tan(fitted[3]) * 1500, fitted[1] - 1/np.tan(fitted[2]) * 1500, 1500]]
    points = [
        [fitted[0], fitted[1], 0],
        [top[0], top[1], top[2]]
    ]
    lines = [[0, 1]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(line_set)

    # line = np.linspace(0, 1000, 11, dtype=np.float64)
    # points = np.meshgrid(line,line,line,indexing='ij')

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=fitted[4],
        height=10000,
        resolution=10
    )
    cylinder.translate([fitted[0], fitted[1], 0])
    cylinder.transform(transform)

    cylinder = o3d.geometry.LineSet.create_from_triangle_mesh(cylinder)
    cylinder.paint_uniform_color([0, 1, 0])
    geometries.append(cylinder)

    # points = []
    # colors = []
    # for x in line:
    #     for y in line:
    #         for z in line:
    #             points.append([x,y,z])
    #             colors.append([x/1000, y/1000, z/1000])

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)

    # shoulder_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5, resolution=20)
    # shoulder_sphere.translate(shoulder_point)

    # elbow_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5, resolution=20)
    # elbow_sphere.translate(elbow_point)

    o3d.visualization.draw_geometries(geometries)
    # o3d.visualization.draw_geometries([pcd])

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
    # visualize_frame_3d(depth_frames[frame_id,:,:], frame_id)
    