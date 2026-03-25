# Computer Vision

This repository contains my learning and works on computer vision using python and its libraries.

## Demo

## To-Do
- [ ] Add demo for ArUco marker detection & Homogeneous detectors
- [ ] Document `markers/` folder
- [ ] Document `mobilenetssd/` folder
- [ ] Add support for video file input (not just live feed)
- [ ] Implement measuring sizes of detected objects

## Table of Contents
 - [Project Structure](#project-structure)
 - [Works](#works)
 - [Tech Stack](#tech-stack)
 - [Installation](#installation)
 - [Usage](#usage)
 - [Config](#config)
 - [LICENSE](#license)


### [Project Structure](#project-structure)

```
computer_vision/
├── assets/
├── homogeneous_detectors/
│   ├── homogeneous_body_detector_byHSV.py
│   ├── homogeneous_body_detector_byHSV.py
│   ├── homogeneous_body_detector_meanshift.py
│   └── homogeneous_detector_kmeans_clustering.py
├── markers/
├── mobilenetssd/
├── generate_aruco_marker.py
├── find_aruco_marker.py
├── augment_aruco_marker.py
├── drawcontours_aruco.py
├── homogeneous_detector.py
├── object_detector.py
├── mobilenet_torch_detector.py
├── .gitignore
├── LICENSE.md
└── README.md
```

### Works
This repository includes my works with:
- ArUco markers
- Homogeneous body detectors
    - HSV based grouping
    - Mean Shift grouping
    - K-Means Color grouping
- MobileNetSSD base detectors:
    - Predefined Classes detection using caffemodel
    - PyTorch based predefined classes detection

### [Tech Stack](#tech-stack)

- Python 3.12.2
- Numpy
- PyTorch
- OpenCV

### Installation
To install the required libraries/dependencies for this repository:
```bash
pip install -r requirements.txt
```

### Usage
All the scripts on this repo are designed to be modular, meaning you can simply import the functions (or the classes) and use them in your code.

Example: 

**With a camera feed:**

```python
import cv2
from homogeneous_detector import HomogeneousDetector

detector = HomogeneousDetector()
cap = cv2.VideoCapture(0)

def main():
    success, img = cap.read()

    if not success:
        return

    while True:
        img = detector.kmeans_detect(img)

        cv2.imshow("K-Means Detected", img)
        cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

**With an Image:**

```python
import cv2
from homogeneous_detector import HomogeneousDetector

detector = HomogeneousDetector()

img = cv2.imread("path/to/image.jpg")
img = detector.kmeans_detect(img)

cv2.imshow("K-Means Detected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```


### Config
Most or all of the scripts were designed with the intent of use with a live camera feed/capture but can also be used on images.


### LICENSE

Distributed under the MIT License. See `LICENSE` for more information.