# Enhanced Vision-Based Detection System for Urban Crosswalks

## Project Overview

This project aims to enhance pedestrian and cyclist safety at urban crosswalks through the use of advanced machine vision technologies. By implementing real-time detection and identification, the system improves driver awareness, aids traffic management systems in preventing accidents, and streamlines traffic flow.

## Project Goals

The primary goal is to create a robust detection system that can accurately identify pedestrians and cyclists in urban environments from both the driver’s perspective and an overhead view. The project also addresses privacy concerns by incorporating face-blurring functionalities to protect individuals' identities in recorded video footage.

## Importance of the Features

In 2023, over 7,500 pedestrians were killed in car crashes across the United States. This project aims to mitigate such incidents by ensuring pedestrian safety at crosswalks and alerting drivers to slow down when pedestrians or cyclists are detected.

## Technologies Used

- **NVIDIA Jetson Orin Nano:** High-performance processing unit for real-time video processing and deep learning tasks.
- **OpenCV:** Utilized for various computer vision tasks.
- **CUDA:** Enhances GPU processing capabilities for faster performance.
- **YOLO (You Only Look Once):** Deep learning model for precise object detection and recognition.

## Functional Requirements

1. **Pedestrian and Cyclist Detection (Driver View and Overhead View):**
   - **Minimum:** Detects human and cyclist presence in video frames.
   - **Target:** Enhances detection accuracy from different perspectives.
   - **Optimal:** Accurate detection of pedestrians and cyclists in the vehicle’s path.

2. **Face Blurring:**
   - **Minimum:** Blurs faces when detected.
   - **Target:** Blurs multiple faces in real-time.
   - **Optimal:** Ensures privacy without compromising detection capabilities.

3. **Alert System:**
   - **Minimum:** Triggers alert upon detecting a pedestrian.
   - **Target:** Generates alert messages for single or multiple pedestrians.
   - **Optimal:** Provides accurate and timely alerts to prevent potential accidents.

4. **CUDA Implementation:**
   - Enhances processing speed and efficiency for real-time video analysis.

5. **YOLO Models Implementation:**
   - Implements YOLO for high precision and rapid detection.

## System Architecture and Design

The system comprises several components that work together to process video inputs, detect relevant objects, and perform actions based on the detections. Key modules include:
- **playground.cpp:** Main module for overhead detection.
- **playground_driver.cpp:** Handles detection from the driver’s perspective.
- **faceblur.cpp:** Applies Gaussian blur to detected faces.
- **utilities.cpp and utilities.h:** Shared functions for image processing, neural network configurations, and result interpretation.

## Performance Evaluation

The system has been rigorously tested and has shown excellent performance in terms of frame rates and detection accuracy. Key metrics include:
- **Frames Per Second (FPS):** Achieved up to 28-35 FPS, exceeding the promised 8 FPS.
- **Precision and Recall:** High precision and recall rates for both pedestrian and cyclist detection, ensuring minimal false positives and accurate monitoring.

## Challenges and Solutions

The project faced several challenges, including low throughput, failure in model deployment, and hardware constraints. These were overcome by:
- Recompiling OpenCV with CUDA and DNN support.
- Using efficient models like Tiny YOLO for rapid detection tasks.
- Implementing parallel processing and concurrency to enhance performance.

## Performance Evaluation

The system has shown excellent performance in terms of frame rates and detection accuracy. Key metrics include:
- **Frames Per Second (FPS):** Achieved up to 28-35 FPS, exceeding the promised 8 FPS.
- **Precision and Recall:** High precision and recall rates for both pedestrian and cyclist detection, ensuring minimal false positives and accurate monitoring.

### Output Images

Below are some sample output images demonstrating the system's capabilities:

![Overhead Pedestrian Detection](https://github.com/raymardi27/Crosswalk-pedestrian-detection/assets/154280528/0ec663a8-0326-4517-a256-0bc162dfadf8)

*Overhead Pedestrian Detection*

![Driver View Pedestrian Detection](https://github.com/raymardi27/Crosswalk-pedestrian-detection/assets/154280528/a3b40a63-67c1-4a3f-83b4-fd8ea6f9865f)

*Driver View Pedestrian Detection*

![Cyclist Detection](https://github.com/raymardi27/Crosswalk-pedestrian-detection/assets/154280528/e23ed681-0a86-48e5-b9c3-e2b6df301ce0)

*Cyclist Detection*

![Face Blurring](https://github.com/raymardi27/Crosswalk-pedestrian-detection/assets/154280528/20a8a5ec-6134-485e-98d8-8fc66433dde1)

*Face Blurring*

![Alert System](https://github.com/raymardi27/Crosswalk-pedestrian-detection/assets/154280528/ae786051-0a02-4eae-b449-5358656b2c1b)

*Alert System*

These images highlight the successful detection and processing of pedestrians and cyclists, as well as the face blurring functionality and alert system in action.


## Conclusion

The Enhanced Vision-Based Detection System for Urban Crosswalks demonstrates the potential of integrating advanced computer vision and machine learning technologies to improve public safety. The project successfully addresses pedestrian and cyclist safety while ensuring privacy and real-time performance.

## Contributors

- **Rutvik Yamkanmardi**
- **Aadil Shaikh**
- **Om Patil**
- **Pradnya Ghadge**
- **Kimaya Sawant**

## References

- NVIDIA CUDA Documentation
- YOLO Tutorial and Documentation
- Jetson ORIN Documentation
- Various academic papers and online resources (detailed in the report).

## Acknowledgments

- Prof Samuel B. Siewert (CSU Chico)
- Joseph Redmon, Ali Farhadi (YOLO Model)
- The House of Black and White (Hall of Faces Model)

