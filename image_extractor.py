import argparse
import cv2
import numpy as np
import sys

# globals
points = []
image_display = None
window_name = "Image Extractor"

def select_points(event, x, y, flags, param):
    """mouse callback"""
    global points, image_display
    
    # record point
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])
            # draw marker
            cv2.circle(image_display, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
            cv2.imshow(window_name, image_display)

def main():
    global points, image_display

    # setup args
    parser = argparse.ArgumentParser(description="Extract and warp a region from an image.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input image file.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the output image file.")
    parser.add_argument("-r", "--resolution", required=True, help="Output resolution in WxH format (e.g., 800x600).")
    
    args = parser.parse_args()

    # parse resolution
    try:
        res_parts = args.resolution.lower().split('x')
        width = int(res_parts[0])
        height = int(res_parts[1])
    except ValueError:
        print("Error: Resolution must be in the format WxH (for example, 800x600).")
        sys.exit(1)

    # load image
    original_image = cv2.imread(args.input)
    if original_image is None:
        print(f"Error: Could not load the image from '{args.input}'. Please check the path.")
        sys.exit(1)

    # restart loop
    while True:
        # reset state
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

        # wait for points
        restart_selection = False
        while len(points) < 4:
            cv2.imshow(window_name, image_display)
            key = cv2.waitKey(10) & 0xFF
            
            if key == 27:  # esc key check
                restart_selection = True
                break
                
        if restart_selection:
            print("Selection discarded.")
            continue
            
        # process points
        print("4 points selected. Warping the image...")
        
        # float array conversion
        src_points = np.array(points, dtype=np.float32)
        
        # destination points
        dst_points = np.array([
            [0, 0],               # top left
            [width - 1, 0],       # top right
            [width - 1, height - 1], # bottom right
            [0, height - 1]       # bottom left
        ], dtype=np.float32)
        
        # get transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # warp image
        warped_image = cv2.warpPerspective(original_image, matrix, (width, height))
        
        # show result
        result_window = "Warped Result"
        cv2.imshow(result_window, warped_image)
        
        print("Result displayed.")
        print("Press 's' to save the result and exit.")
        print("Press ESC to discard the result and start over.")
        
        # wait for save
        saved = False
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == 27:  # esc key
                cv2.destroyWindow(result_window)
                print("Result discarded.")
                break
                
            elif key == ord('s') or key == ord('S'):  # s key
                # save image
                success = cv2.imwrite(args.output, warped_image)
                if success:
                    print(f"Successfully saved warped image to: {args.output}")
                else:
                    print(f"Error: Failed to save the image to '{args.output}'.")
                saved = True
                break
                
        # exit check
        if saved:
            break

    # cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
