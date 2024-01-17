Gaze Tracker

A program that 
1. Uses the shape_predictor_68_face_landmarks_GTX model to predict the face feature points as shown in parts.PNG.
2. Extracts the eye locations from those face feature points.
3. Applies image processing techniques to extract the pupil and obtain the pupil centre.
4. Compares the extracted pupil with a reference position (obtained in calibration phase) to estimate where the user is looking on the screen.
5. Moving the mouse to the expected look position.

The program is divided into two phases:
1. Calibration phase: the user is instructed to look at the middle of the screen, and their eye position is recorded as a reference point
2. Tracking phase: the user can move their gaze without moving their head and the program moves the mouse accordingly

So far the program is able to identify the general region where the user is looking but is not accurate enough for practical use. 
Furthermore there is a delay between the user looking at a place and the mouse moving to it.
Moreover, the mouse moves chaoticly in the tracking phase, due to the fact that small changes in eye movement mean large changes in the look-at position.

Possible Improvements:
1. Use a model that detects only the eye features to improve performance.
2. Use a model that predicts eyebrow regions, reducing the need for a calibration phase entirely.
3. Use both eyes as a basis for prediction, thus improving accuarcy and lowering stuttering.
