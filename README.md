# ASL Hand Sign Detection


## Overview

This project is a learning-based attempt to build an American Sign Language recognition tool using computer vision and machine learning.

At this stage, I am focusing on understanding hand tracking and landmark data using MediaPipe before moving on to machine learning models.

## Current Progress

So far, I have:

* Set up a Python virtual environment
* Installed `mediapipe`, `opencv-python`, and `numpy`
* Created a webcam script using OpenCV
* Used MediaPipe to detect hand landmarks
* Printed landmark coordinates to the terminal
* Explored how the `x`, `y`, and `z` values change as the hand moves
* Built a simple rule-based detector for open hand vs closed hand

This version is not machine learning yet. It uses simple coordinate comparisons to help me understand how hand landmark data works.

## How It Works

The script uses OpenCV to access the webcam and MediaPipe to detect 21 hand landmarks. Each landmark has an `x`, `y`, and `z` coordinate.

For the current open/closed hand experiment, I compared the `y` values of fingertips and finger joints. I observed that a smaller `y` value means a point is higher on the screen.

## How to Run

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install mediapipe opencv-python numpy
```

Run the script:

```bash
python track_hands.py
```

Press `q` to close the webcam window.

## What I Learned

* MediaPipe does not automatically understand gestures; it provides landmark data
* A hand is represented as 21 landmark points
* Landmark coordinates can be used to create simple gesture rules
* The `y` coordinate works differently from normal graph coordinates
* Rule-based detection is a useful first step before machine learning

## Current Limitations

* Only detects a basic open or closed hand
* Does not recognise ASL letters yet
* May not work well if the hand is rotated or sideways
* Does not use a trained ML model yet

## Next Steps

* Experiment with more rule-based gestures
* Research ASL alphabet datasets
* Train a simple classifier using hand landmark features
* Connect the model to webcam input
* Build a simple web interface

## Status

Current phase: Foundation and landmark exploration.
