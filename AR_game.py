import cv2
import numpy as np
import pyglet
from pyglet import shapes
import mediapipe as mp
import random
import math

# config
# hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# aruco config
# use dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# camera
cap = cv2.VideoCapture(0)
cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if cam_width == 0 or cam_height == 0:
    cam_width, cam_height = 640, 480 # fallback

print(f"Camera Resolution: {cam_width}x{cam_height}")

# pyglet setup
# get screen size
display = pyglet.display.get_display()
screen = display.get_default_screen()
screen_width = screen.width
screen_height = screen.height

print(f"Screen Resolution: {screen_width}x{screen_height}")

# open window
window = pyglet.window.Window(screen_width, screen_height, "AR Target Destroyer", resizable=True)
batch = pyglet.graphics.Batch()

# toggle fullscreen
@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.F:
        window.set_fullscreen(not window.fullscreen)


# game state
targets = []
score = 0
finger_pos = None # finger position
background_image = None
spawn_timer = 0.0
markers_detected = False

def order_points(pts):
    """order corners"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # top left
    rect[2] = pts[np.argmax(s)] # bottom right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top right
    rect[3] = pts[np.argmax(diff)] # bottom left

    return rect

def spawn_target():
    """spawn target"""
    x = random.randint(50, window.width - 50)
    y = window.height + 50 # spawn high
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

    # spawn check
    spawn_timer += dt
    if spawn_timer > 1.0:
        if markers_detected:
            spawn_target()
        spawn_timer = 0.0

    # move down
    for t in targets:
        t['y'] -= t['speed'] * dt

    # remove off screen
    targets[:] = [t for t in targets if t['y'] + t['radius'] > 0]

    # check collision
    if finger_pos is not None:
        fx, fy = finger_pos
        surviving_targets = []
        for t in targets:
            dist = math.hypot(t['x'] - fx, t['y'] - fy)
            if dist < t['radius'] + 15: # hit check
                score += 10 # score up
                # effect slot
            else:
                surviving_targets.append(t)
        
        targets[:] = surviving_targets

def update_camera(dt):
    global background_image, finger_pos, markers_detected
    
    ret, frame = cap.read()
    if not ret:
        return

    # detect aruco
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco_detector.detectMarkers(gray)

    display_frame = frame.copy()
    M = None

    if ids is not None and len(ids) >= 4:
        markers_detected = True
        # get centers
        centers = []
        for i in range(4):
            corner = corners[i][0]
            center_x = int(corner[:, 0].mean())
            center_y = int(corner[:, 1].mean())
            centers.append([center_x, center_y])
        
        pts = np.array(centers, dtype="float32")
        rect = order_points(pts)

        # warp destinations
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
        # draw markers
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
        # fallback mirror
        display_frame = cv2.flip(display_frame, 1)

    # hand track
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    finger_pos = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # index tip
            tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            if markers_detected and M is not None:
                # pixel coords
                raw_x = tip.x * cam_width
                raw_y = tip.y * cam_height
                
                # transform point
                point = np.array([[[raw_x, raw_y]]], dtype="float32")
                transformed_point = cv2.perspectiveTransform(point, M)
                tx, ty = transformed_point[0][0]
                
                # map to window
                px = int((tx / cam_width) * window.width)
                py = int((1.0 - (ty / cam_height)) * window.height)
            else:
                # mirror map
                px = int((1.0 - tip.x) * window.width)
                py = int((1.0 - tip.y) * window.height)
            
            finger_pos = (px, py)
            break # track one

    # convert to pyglet
    # invert vertical
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
    
    # draw webcam
    if background_image is not None:
        background_image.blit(0, 0, width=window.width, height=window.height)
        
    # reset shapes
    drawn_shapes = []

    # draw targets
    for t in targets:
        circle = shapes.Circle(x=t['x'], y=t['y'], radius=t['radius'], 
                               color=t['color'], batch=batch)
        drawn_shapes.append(circle)
        
    # draw finger
    if finger_pos is not None:
        fx, fy = finger_pos
        finger_circle = shapes.Circle(x=fx, y=fy, radius=15, 
                                      color=(255, 50, 50), batch=batch)
        drawn_shapes.append(finger_circle)
        
    # draw score
    score_label = pyglet.text.Label(f'Score: {score}',
                          font_name='Arial',
                          font_size=24,
                          x=20, y=window.height - 40,
                          color=(255, 255, 255, 255),
                          batch=batch)
    drawn_shapes.append(score_label)
    
    if len(targets) > 0 or finger_pos is not None or True:
        batch.draw()

# schedule
# logic rate
pyglet.clock.schedule_interval(update_camera, 1/60.0)
pyglet.clock.schedule_interval(update_game_logic, 1/60.0)

if __name__ == '__main__':
    print("Starting AR Game...")
    print("Please hold 4 ArUco markers up to the camera to define the play area.")
    pyglet.app.run()
    
    # cleanup
    cap.release()
    cv2.destroyAllWindows()
