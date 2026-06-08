import cv2 
import numpy as np 
import pyglet 
from pyglet import shapes 
from pyglet .window import key 
import sys 
import math 

from DIPPID import SensorUDP 

# configuration
PORT =5700 
ACCEL_SCALAR =5.0 # multiplier for accelerometer data to convert to pixels
TARGET_IDS =[5 ,23 ]# accepted smartphone aruco marker ids

# global state
alpha =0.5 
cam_pos =None # x y from camera
pred_pos =None # x y fused prediction

background_image =None 
markers_detected =False 
M =None # perspective transform matrix

# aruco setup
aruco_dict =cv2 .aruco .getPredefinedDictionary (cv2 .aruco .DICT_6X6_250 )
aruco_params =cv2 .aruco .DetectorParameters ()
aruco_detector =cv2 .aruco .ArucoDetector (aruco_dict ,aruco_params )

# camera setup
import sys 
if sys .platform =='win32':
    cap =cv2 .VideoCapture (0 ,cv2 .CAP_DSHOW )
else :
    cap =cv2 .VideoCapture (0 )
cam_width =int (cap .get (cv2 .CAP_PROP_FRAME_WIDTH ))
cam_height =int (cap .get (cv2 .CAP_PROP_FRAME_HEIGHT ))
if cam_width ==0 or cam_height ==0 :
    cam_width ,cam_height =640 ,480 
print (f"Camera Resolution: {cam_width }x{cam_height }")

# dippid setup
print ("Initializing DIPPID...")
sensor =SensorUDP (PORT )

# pyglet window setup
print ("Initializing Pyglet window...")
display =pyglet .display .get_display ()
screen =display .get_default_screen ()
window =pyglet .window .Window (cam_width ,cam_height ,"Sensor Fusion",resizable =True )
batch =pyglet .graphics .Batch ()

print ("Initialization complete.")
print ("Starting Sensor Fusion...")
print ("Please hold 4 ArUco markers up to the camera to define the board.")
print ("Use a smartphone with ArUco ID 5 or 23 for tracking.")
print ("Use Arrow UP/DOWN to adjust Alpha.")

def order_points (pts ):
    rect =np .zeros ((4 ,2 ),dtype ="float32")
    s =pts .sum (axis =1 )
    rect [0 ]=pts [np .argmin (s )]
    rect [2 ]=pts [np .argmax (s )]

    diff =np .diff (pts ,axis =1 )
    rect [1 ]=pts [np .argmin (diff )]
    rect [3 ]=pts [np .argmax (diff )]
    return rect 

def update_game_logic (dt ):
    global pred_pos ,cam_pos ,alpha 

    # 1 read button 1 to reset
    btn1 =sensor .get_value ('button_1')
    if btn1 ==1 :
        print ("DEBUG DIPPID: Button 1 pressed")
        pred_pos =cam_pos # reset to camera

    if not hasattr (update_game_logic ,'debug_timer'):
        update_game_logic .debug_timer =0.0 
    update_game_logic .debug_timer +=dt 

    # 2 read accelerometer
    accel =sensor .get_value ('accelerometer')
    if accel is not None and update_game_logic .debug_timer >0.5 :
        print (f"DEBUG DIPPID Accelerometer: {accel }")
        update_game_logic .debug_timer =0.0 

    accel_x ,accel_y =0.0 ,0.0 
    if accel is not None and isinstance (accel ,dict ):
    # depending on phone orientation you may need to
        accel_x =accel .get ('x',0.0 )
        accel_y =accel .get ('y',0.0 )

        # 3 apply complementary filter
    if cam_pos is not None :
        if pred_pos is None :
            pred_pos =cam_pos 

            # complementary filter logic
            # predpos alpha predpos accel dt 1alpha campos
            # note accelerometer y might need inversion depending on
        new_x =alpha *(pred_pos [0 ]+accel_x *dt *ACCEL_SCALAR *window .width )+(1 -alpha )*cam_pos [0 ]
        new_y =alpha *(pred_pos [1 ]+accel_y *dt *ACCEL_SCALAR *window .height )+(1 -alpha )*cam_pos [1 ]

        pred_pos =(new_x ,new_y )

def update_camera (dt ):
    global background_image ,cam_pos ,markers_detected ,M 

    ret ,frame =cap .read ()
    if not ret :
        return 

    gray =cv2 .cvtColor (frame ,cv2 .COLOR_BGR2GRAY )
    corners ,ids ,rejected =aruco_detector .detectMarkers (gray )

    display_frame =frame .copy ()

    board_corners =[]
    phone_center =None 

    if ids is not None :
        for i ,m_id in enumerate (ids ):
            m_id =m_id [0 ]
            if m_id in TARGET_IDS :
                c =corners [i ][0 ]
                cx =int (c [:,0 ].mean ())
                cy =int (c [:,1 ].mean ())
                phone_center =(cx ,cy )
            else :
                if len (board_corners )<4 :
                    c =corners [i ][0 ]
                    cx =int (c [:,0 ].mean ())
                    cy =int (c [:,1 ].mean ())
                    board_corners .append ([cx ,cy ])

    if len (board_corners )==4 :
        markers_detected =True 
        pts =np .array (board_corners ,dtype ="float32")
        rect =order_points (pts )

        dst =np .array ([
        [0 ,0 ],
        [cam_width -1 ,0 ],
        [cam_width -1 ,cam_height -1 ],
        [0 ,cam_height -1 ]
        ],dtype ="float32")

        M =cv2 .getPerspectiveTransform (rect ,dst )
        display_frame =cv2 .warpPerspective (frame ,M ,(cam_width ,cam_height ))
    else :
        markers_detected =False 
        if ids is not None :
            cv2 .aruco .drawDetectedMarkers (display_frame ,corners ,ids )
        display_frame =cv2 .flip (display_frame ,1 )

        # process phone tracking
    cam_pos =None 
    if phone_center is not None :
        if markers_detected and M is not None :
            raw_x ,raw_y =phone_center 
            point =np .array ([[[raw_x ,raw_y ]]],dtype ="float32")
            transformed =cv2 .perspectiveTransform (point ,M )
            tx ,ty =transformed [0 ][0 ]

            px =int ((tx /cam_width )*window .width )
            py =int ((1.0 -(ty /cam_height ))*window .height )
            cam_pos =(px ,py )
        else :
            rx ,ry =phone_center 
            px =int ((1.0 -(rx /cam_width ))*window .width )
            py =int ((1.0 -(ry /cam_height ))*window .height )
            cam_pos =(px ,py )

    display_frame_flipped =cv2 .flip (display_frame ,0 )
    rgb_flipped =cv2 .cvtColor (display_frame_flipped ,cv2 .COLOR_BGR2RGB )

    pitch =cam_width *3 
    image_data =pyglet .image .ImageData (
    cam_width ,cam_height ,'RGB',rgb_flipped .tobytes (),pitch =pitch 
    )
    background_image =image_data 

@window .event 
def on_draw ():
    window .clear ()

    if background_image is not None :
        background_image .blit (0 ,0 ,width =window .width ,height =window .height )

    drawn_shapes =[]

    if cam_pos is not None :
        cx ,cy =cam_pos 
        red_dot =shapes .Circle (x =cx ,y =cy ,radius =15 ,color =(255 ,0 ,0 ),batch =batch )
        drawn_shapes .append (red_dot )

    if pred_pos is not None :
        px ,py =pred_pos 
        green_dot =shapes .Circle (x =px ,y =py ,radius =10 ,color =(0 ,255 ,0 ),batch =batch )
        drawn_shapes .append (green_dot )

    alpha_label =pyglet .text .Label (f'Alpha: {alpha :.2f}',
    font_name ='Arial',font_size =20 ,
    x =20 ,y =window .height -30 ,
    color =(255 ,255 ,255 ,255 ),batch =batch )
    drawn_shapes .append (alpha_label )

    batch .draw ()

@window .event 
def on_key_press (symbol ,modifiers ):
    global alpha 
    if symbol ==key .Q :
        window .close ()
        pyglet .app .exit ()
    elif symbol ==key .UP :
        alpha =min (1.0 ,alpha +0.05 )
    elif symbol ==key .DOWN :
        alpha =max (0.0 ,alpha -0.05 )

@window .event 
def on_close ():
    window .close ()
    pyglet .app .exit ()

pyglet .clock .schedule_interval (update_camera ,1 /60.0 )
pyglet .clock .schedule_interval (update_game_logic ,1 /60.0 )

if __name__ =='__main__':
    try :
        pyglet .app .run ()
    finally :
        try :
            if hasattr (sensor ,'disconnect'):
                sensor .disconnect ()
            elif hasattr (sensor ,'stop'):
                sensor .stop ()
        except :
            pass 
        cap .release ()
        cv2 .destroyAllWindows ()
        import os 
        os ._exit (0 )
