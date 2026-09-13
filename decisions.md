Rewrite it as a list of choices you made, each with the reason. Roughly this shape, but in your words and with your reasoning:

Scope. 24 letters, J and Z excluded because they're signed with movement and a single frame can't capture that.

Handedness. MediaPipe reports Left or Right per detection. Stored as a column. Kaggle is mostly Right, my own frames are Left, so they need reconciling before evaluation. Plan: mirror x to normalise everything to one handedness.

Sampling. Every 10th image rather than the first N, because the source images are consecutive video frames and the first 300 are all near-identical.

Evaluation split. Training on Kaggle, testing on my own webcam frames, rather than a random split of the Kaggle data. A random split would put near-duplicates on both sides and inflate the accuracy.

Detection rates. Per letter, 40% for N up to 96% for F. Closed-fist letters detect worst. This leaves the classes imbalanced.

Environment. Python 3.9.6 with mediapipe 0.10.14 via --no-deps, because 1.0.1 on Python 3.14 crashes on macOS Metal init and jax won't compile on 3.9. jax isn't needed for hand landmarking.

CSV format. 63 landmark values, handedness, label. Label last so "everything but the final column" is the feature set.

Normalisation - Two transforms on the training data, one for wrist-relative positions and scale.

Wrist-relative: subtract landmark 0's x, y, z from every landmark. Raw coordinates encode where the hand is in frame, and KNN compares numbers directly, so without this the model keys on position rather than shape. After the subtraction each value is a distance from the wrist, which doesn't change when the hand moves around the frame.

Scale: divide by the wrist-to-landmark-9 distance. Subtracting fixes position but not apparent size, which varies with distance from camera. Dividing by a reference distance from the same hand cancels that out. (Essentially Right now a hand near the camera produces bigger numbers than the same hand further away. Dividing by a reference distance from the same hand cancels that.). 
    Wrist to landmark 9 because it's a palm-anchored distance that doesn't change with the sign being made

Handedness mirroring: x becomes 1-x for right hands, so everything is normalised to left. Needed because the training data is mostly right hands and my test frames are left.

All three live in landmarks.py so the notebook and the Flask app can't drift apart.