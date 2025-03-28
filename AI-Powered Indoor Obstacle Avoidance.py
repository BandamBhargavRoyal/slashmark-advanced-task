!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import tensorflow as tf
import cv2
import numpy as np
import heapq

# Load the pre-trained TensorFlow model
model = tf.saved_model.load("path/to/saved_model")
bridge = CvBridge()

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(start, goal, grid):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < grid.shape[0]:
                if 0 <= neighbor[1] < grid.shape[1]:
                    if grid[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    continue
            else:
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return False

def detect_obstacles(image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    threshold = 0.5
    valid_detections = boxes[scores > threshold]
    return valid_detections

def lidar_callback(data):
    # Placeholder for LIDAR data processing
    rospy.loginfo("LIDAR data received")

def camera_callback(data):
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    obstacles = detect_obstacles(cv_image)
    rospy.loginfo("Detected obstacles: %s", obstacles)

    # Create an occupancy grid
    grid = np.zeros((100, 100))
    for box in obstacles:
        # Convert box coordinates to grid indices
        x1, y1, x2, y2 = box
        grid[int(y1*100):int(y2*100), int(x1*100):int(x2*100)] = 1

    # Plan path avoiding obstacles
    start = (0, 0)
    goal = (99, 99)
    path = a_star(start, goal, grid)
    rospy.loginfo("Path: %s", path)

    # Control robot to follow the path
    if path:
        twist = Twist()
        # Example: Move forward
        twist.linear.x = 0.1
        pub_cmd.publish(twist)

if __name__ == '__main__':
    rospy.init_node('obstacle_avoidance_node', anonymous=True)

    # Publishers and Subscribers
    pub_cmd = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    rospy.Subscriber('scan', LaserScan, lidar_callback)
    rospy.Subscriber('camera/rgb/image_raw', Image, camera_callback)

    rospy.spin()
