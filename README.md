# GazeTracker

GazeTracker is a simple computer-vision program that moves the mouse cursor based on the user's eye movement.

The program uses a webcam to detect the user's face and locate the eyes using a **pretrained dlib 68-point facial landmark model**. It then extracts the left-eye region and uses several digital image-processing techniques to detect the pupil. The pupil's position is compared with a calibrated reference position, and the resulting displacement is used to control the mouse cursor.

## How It Works

The program follows these main steps:

1. **Face Detection**
   A webcam frame is processed using dlib's frontal face detector to locate the user's face.

2. **Facial Landmark Detection**
   The pretrained `shape_predictor_68_face_landmarks_GTX.dat` model is used to identify facial landmarks, including the points surrounding the eyes.

3. **Eye Region Extraction**
   The six landmarks surrounding the left eye are used to crop a smaller image containing the eye.

4. **Pupil Detection**
   The eye image is processed using OpenCV:

   * Grayscale conversion
   * Histogram equalization
   * Binary thresholding
   * Erosion
   * Dilation
   * Median blurring
   * Blob detection

   The detected blob is used as an estimate of the pupil's centre.

5. **Calibration**
   The user looks at the centre of the webcam display and presses **Enter**. The current pupil position is stored as the calibrated reference point.

6. **Mouse Control**
   As the pupil moves away from the calibrated position, the displacement is scaled and translated into screen coordinates. A short moving average is applied to the mouse positions to reduce jitter before moving the cursor.

## Machine Learning Model

The project uses a **pretrained dlib facial landmark model**:

`shape_predictor_68_face_landmarks_GTX.dat`

This model predicts 68 facial landmark points from a detected face. It is used here to locate the eye rather than directly predicting the user's gaze.

The pupil detection and gaze calculation are implemented separately using traditional image-processing and geometric techniques. Therefore, the project combines a pretrained machine-learning model with digital image processing rather than using an end-to-end gaze-estimation neural network.

## Requirements

* Python 3
* Webcam
* `OpenCV`
* `dlib`
* `NumPy`
* `PyAutoGUI`
* The pretrained dlib 68-point facial landmark model

Install the Python dependencies with:

```bash
pip install opencv-python dlib numpy pyautogui
```

The facial landmark model file must be placed in the project directory:

```text
shape_predictor_68_face_landmarks_GTX.dat
```

## How to Use

Run the program:

```bash
python main.py
```

When the webcam window appears:

1. Make sure your face and eye are well lit.
2. Keep your head relatively still.
3. Look at the centre of the webcam display.
4. Press **Enter** to calibrate.
5. Move your eyes to control the mouse cursor.
6. Press **Q** to exit.

For more reliable tracking, avoid large head movements after calibration and make sure the pupil is clearly visible to the camera.

## Limitations

This is a simple gaze-tracking implementation rather than a full gaze-estimation system. Its accuracy depends on factors such as lighting, camera position, pupil visibility, head movement, and the thresholding parameters used for pupil detection.

So far the program is able to identify the general region where the user is looking but is not accurate enough for practical use.
Furthermore there is a delay between the user looking at a place and the mouse moving to it.
Moreover, the mouse moves chaoticly in the tracking phase, due to the fact that small changes in eye movement mean large changes in the look-at position.

## Possible Improvements

1. Use a model that detects only the eye features, instead of all face features, to improve performance.
2. Use a model that predicts eyebrow regions, reducing the need for a calibration phase entirely.
3. Use both eyes as a basis for prediction, thus improving accuarcy and lowering stuttering.
