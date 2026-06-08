import cv2
import numpy as np
import pyglet
from pyglet import shapes
import mediapipe as mp
import random
import math

import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'hand_landmarker.task')

# configuration settings
# hand tracking setup
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
hands = vision.HandLandmarker.create_from_options(options)

# aruco configuration
# use predefined dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# camera setup
import sys
if sys.platform == 'win32':
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(0)

cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if cam_width == 0 or cam_height == 0:
    cam_width, cam_height = 640, 480 # fallback resolution

print(f"Camera Resolution: {cam_width}x{cam_height}")

# setup pyglet
# retrieve screen size
display = pyglet.display.get_display()
screen = display.get_default_screen()
screen_width = screen.width
screen_height = screen.height

print(f"Screen Resolution: {screen_width}x{screen_height}")

# create window
window = pyglet.window.Window(screen_width, screen_height, "AR Target Destroyer", resizable=True)
batch = pyglet.graphics.Batch()

# fullscreen toggle event
@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.F:
        window.set_fullscreen(not window.fullscreen)
    elif symbol == pyglet.window.key.Q:
        window.close()
        pyglet.app.exit()


# game variables
targets = []
score = 0
finger_pos = None # current finger position
background_image = None
spawn_timer = 0.0
markers_detected = False

def order_points(pts):
    """sort rectangle corners"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # point top left
    rect[2] = pts[np.argmax(s)] # point bottom right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # point top right
    rect[3] = pts[np.argmax(diff)] # point bottom left

    return rect

def spawn_target():
    """create new target"""
    x = random.randint(50, window.width - 50)
    y = window.height + 50 # spawn above screen
    radius = random.randint(20, 40)
    speed = random.uniform(100, 250)
    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    targets.append({
        'x': x, 'y': y, 
        'radius': radius, 
        'speed': speed, 
        'color': color
    })

def update_game_logic(dt):
    global score, finger_pos, spawn_timer

    # check spawn timer
    spawn_timer += dt
    if spawn_timer > 1.0:
        if markers_detected:
            spawn_target()
        spawn_timer = 0.0

    # move targets down
    for t in targets:
        t['y'] -= t['speed'] * dt

    # remove invisible targets
    targets[:] = [t for t in targets if t['y'] + t['radius'] > 0]

    # check for hits
    if finger_pos is not None:
        fx, fy = finger_pos
        surviving_targets = []
        for t in targets:
            dist = math.hypot(t['x'] - fx, t['y'] - fy)
            if dist < t['radius'] + 15: # collision detected
                score += 10 # increase game score
                # particle effect slot
            else:
                surviving_targets.append(t)
        
        targets[:] = surviving_targets

def update_camera(dt):
    global background_image, finger_pos, markers_detected
    
    ret, frame = cap.read()
    if not ret:
        return

    # find aruco markers
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco_detector.detectMarkers(gray)

    display_frame = frame.copy()
    M = None

    if ids is not None and len(ids) >= 4:
        markers_detected = True
        # calculate marker centers
        centers = []
        for i in range(4):
            corner = corners[i][0]
            center_x = int(corner[:, 0].mean())
            center_y = int(corner[:, 1].mean())
            centers.append([center_x, center_y])
        
        pts = np.array(centers, dtype="float32")
        rect = order_points(pts)

        # destination coordinates
        dst = np.array([
            [0, 0],
            [cam_width - 1, 0],
            [cam_width - 1, cam_height - 1],
            [0, cam_height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        display_frame = cv2.warpPerspective(frame, M, (cam_width, cam_height))
    else:
        markers_detected = False
        # highlight found markers
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
        # mirror camera view
        display_frame = cv2.flip(display_frame, 1)

    # track user hands
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(time.time() * 1000)
    results = hands.detect_for_video(mp_image, timestamp_ms)

    finger_pos = None
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # index finger tip
            tip = hand_landmarks[8]
            
            if markers_detected and M is not None:
                # raw pixel coordinates
                raw_x = tip.x * cam_width
                raw_y = tip.y * cam_height
                
                # apply perspective transform
                point = np.array([[[raw_x, raw_y]]], dtype="float32")
                transformed_point = cv2.perspectiveTransform(point, M)
                tx, ty = transformed_point[0][0]
                
                # map to screen
                px = int((tx / cam_width) * window.width)
                py = int((1.0 - (ty / cam_height)) * window.height)
            else:
                # mirrored screen map
                px = int((1.0 - tip.x) * window.width)
                py = int((1.0 - tip.y) * window.height)
            
            finger_pos = (px, py)
            break # only track one

    # prepare pyglet image
    # flip image vertically
    display_frame_flipped = cv2.flip(display_frame, 0)
    rgb_flipped = cv2.cvtColor(display_frame_flipped, cv2.COLOR_BGR2RGB)
    
    pitch = cam_width * 3
    image_data = pyglet.image.ImageData(
        cam_width, cam_height, 'RGB', rgb_flipped.tobytes(), pitch=pitch
    )
    background_image = image_data

@window.event
def on_draw():
    window.clear()
    
    # render webcam background
    if background_image is not None:
        background_image.blit(0, 0, width=window.width, height=window.height)
        
    # clear old shapes
    drawn_shapes = []

    # render falling targets
    for t in targets:
        circle = shapes.Circle(x=t['x'], y=t['y'], radius=t['radius'], 
                               color=t['color'], batch=batch)
        drawn_shapes.append(circle)
        
    # render finger position
    if finger_pos is not None:
        fx, fy = finger_pos
        finger_circle = shapes.Circle(x=fx, y=fy, radius=15, 
                                      color=(255, 50, 50), batch=batch)
        drawn_shapes.append(finger_circle)
        
    # render current score
    score_label = pyglet.text.Label(f'Score: {score}',
                          font_name='Arial',
                          font_size=24,
                          x=20, y=window.height - 40,
                          color=(255, 255, 255, 255),
                          batch=batch)
    drawn_shapes.append(score_label)
    
    if len(targets) > 0 or finger_pos is not None or True:
        batch.draw()

@window.event
def on_close():
    window.close()
    pyglet.app.exit()

# schedule update loops
# set logic rate
pyglet.clock.schedule_interval(update_camera, 1/60.0)
pyglet.clock.schedule_interval(update_game_logic, 1/60.0)

if __name__ == '__main__':
    print("Starting AR Game...")
    print("Please hold 4 ArUco markers up to the camera to define the play area.")
    try:
        pyglet.app.run()
    finally:
        # release camera resources
        cap.release()
        cv2.destroyAllWindows()
        import sys
        sys.exit(0)
