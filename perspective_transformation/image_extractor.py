import argparse
import cv2
import numpy as np
import sys

# global variables
points = []
image_display = None
window_name = "Image Extractor"

def select_points(event, x, y, flags, param):
    """handle mouse events"""
    global points, image_display
    
    # save click coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])
            # draw green circle
            cv2.circle(image_display, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
            cv2.imshow(window_name, image_display)

def main():
    global points, image_display

    # configure command line arguments
    parser = argparse.ArgumentParser(description="Extract and warp a region from an image.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input image file.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the output image file.")
    parser.add_argument("-r", "--resolution", required=True, help="Output resolution in WxH format (e.g., 800x600).")
    
    args = parser.parse_args()

    # extract width and height
    try:
        res_parts = args.resolution.lower().split('x')
        width = int(res_parts[0])
        height = int(res_parts[1])
    except ValueError:
        print("Error: Resolution must be in the format WxH (for example, 800x600).")
        sys.exit(1)

    # read image from disk
    original_image = cv2.imread(args.input)
    if original_image is None:
        print(f"Error: Could not load the image from '{args.input}'. Please check the path.")
        sys.exit(1)

    # loop until successful warp
    while True:
        # reset points and image
        points = []
        image_display = original_image.copy()
        
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, select_points)
        
        print("\n--- New Selection ---")
        print("Please click 4 points in the image in the following order:")
        print("1. Top-Left")
        print("2. Top-Right")
        print("3. Bottom-Right")
        print("4. Bottom-Left")
        print("Press ESC at any time to discard changes and start over.")
        print("Press 'q' to quit.")

        # loop until four clicks
        restart_selection = False
        while len(points) < 4:
            cv2.imshow(window_name, image_display)
            key = cv2.waitKey(10) & 0xFF
            
            if key == 27:  # check if escape pressed
                restart_selection = True
                break
            elif key == ord('q') or key == ord('Q'):  # user pressed q
                print("Quitting application.")
                return
                
        if restart_selection:
            print("Selection discarded.")
            continue
            
        # warp based on clicks
        print("4 points selected. Warping the image...")
        
        # convert points to float32
        src_points = np.array(points, dtype=np.float32)
        
        # set destination image corners
        dst_points = np.array([
            [0, 0],               # top left
            [width - 1, 0],       # top right
            [width - 1, height - 1], # bottom right
            [0, height - 1]       # bottom left
        ], dtype=np.float32)
        
        # compute perspective transform
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # apply perspective transform
        warped_image = cv2.warpPerspective(original_image, matrix, (width, height))
        
        # display the warped image
        result_window = "Warped Result"
        cv2.imshow(result_window, warped_image)
        
        print("Result displayed.")
        print("Press 's' to save the result and exit.")
        print("Press ESC to discard the result and start over.")
        print("Press 'q' to quit.")
        
        # loop until save decision
        saved = False
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == 27:  # user pressed escape
                cv2.destroyWindow(result_window)
                print("Result discarded.")
                break
            elif key == ord('q') or key == ord('Q'):  # user pressed q
                print("Quitting application.")
                return
                
            elif key == ord('s') or key == ord('S'):  # user pressed s
                # write image to disk
                success = cv2.imwrite(args.output, warped_image)
                if success:
                    print(f"Successfully saved warped image to: {args.output}")
                else:
                    print(f"Error: Failed to save the image to '{args.output}'.")
                saved = True
                break
                
        # break if image saved
        if saved:
            break

    # close all opencv windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
