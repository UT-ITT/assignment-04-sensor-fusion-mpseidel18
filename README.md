[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/AktWbCri)
# assignment-04-CV-Sensor-Fusion
This repository contains two main programs: an AR-based target destroyer game and a perspective warping image extractor tool.

___

## 1. Perspective Image Extractor (`image_extractor.py`)

A utility tool to extract and warp perspective regions from an input image to a target resolution by manually selecting four corners.

### Execution

```bash
python image_extractor.py -i <input_image> -o <output_image> -r <width>x<height>
```

**Example:**
```bash
python image_extractor.py -i sampleImage.jpg -o sampleImage_resized.jpg -r 800x800
```

### Usage Instructions
1. Left-click four points in the image in the exact order:
   - **Point 1:** Top-Left
   - **Point 2:** Top-Right
   - **Point 3:** Bottom-Right
   - **Point 4:** Bottom-Left
2. Press **`ESC`** at any point to discard selection and start over.
3. Review the preview window:
   - Press **`S`** to save the warped image and exit.
   - Press **`ESC`** to discard the warped result and re-select the points.
     
Using my sample image was not necessary, but it looked funny

---

## 2. AR Target Destroyer (`AR_game.py`)

An interactive AR game where falling targets can be destroyed by tracking your index finger tip. The play area is mapped and transformed when 4 ArUco markers are detected.

### Controls and Features
- **ArUco Markers:** Set up with the default classroom `DICT_6X6_250` dictionary. Hold 4 markers in view of the camera.
- **Dynamic Warping:** Once all 4 markers are detected, the play area is warped to the screen. 
- **Target Spawning:** Targets only spawn when all 4 markers are successfully detected.
- **Visual Outlines:** If markers are partially detected (fewer than 4), green boundary outlines highlight them.
- **Responsive Sizing:** The window opens in your screen's native resolution. Press **`F`** to toggle fullscreen mode.
- **Reliable Hand Tracking:** MediaPipe processes hand tracking on the raw unwarped frame and projects the coordinates, preventing tracking failures from warped image distortion.

## 3. Sensorfusion (`sensor_fusion.py`)

An application that tracks a smartphone using camera vision and fuses it with accelerometer data via a complementary filter to provide smooth tracking.

### Controls and Features
- **ArUco Markers:** Set up with the default classroom `DICT_6X6_250` dictionary. Hold 4 markers in view of the camera to define the tracking area.
- **Dynamic Warping:** Once all 4 markers are detected, the board area is warped to the screen.
- **Visual Outlines:** If markers are partially detected (fewer than 4), boundary outlines highlight them.
- **Smartphone Tracking:** Uses ArUco ID `5` or `23` to track a smartphone's position (indicated by a red dot).
- **Sensor Fusion (Complementary Filter):** Fuses the smartphone's accelerometer data (received via DIPPID) with the camera tracking to predict and smooth movement (indicated by a green dot).
- **Alpha Adjustment:** Press the **`UP`** and **`DOWN`** arrow keys to adjust the Alpha value of the complementary filter, balancing between camera and accelerometer data.
- **Reset Position:** Press **Button 1** on the DIPPID app to reset the fused prediction position back to the raw camera position.
- **Quit:** Press **`Q`** to safely close the application.

## Credits
- Gemini for formatting, sort of proofreading this README and debugging
- OpenCV Documentation
