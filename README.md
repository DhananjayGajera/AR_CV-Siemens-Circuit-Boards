# AR_CV-Siemens-Circuit-Boards 
## Overview
This project implements a component detection system for an AR application using the **YOLOv11** model. The objective is to accurately detect and identify various  components within a circuit Boards, enhancing the interactivity and realism of the AR experience. The system is designed to recognize and label multiple components in real-time using image data, ensuring high accuracy and performance.

## Why YOLOv11?
After extensive experimentation with different algorithms, **YOLOv11** was selected due to its superior detection accuracy, real-time performance, and ease of integration. 

- Initially, the **SIFT algorithm** was explored but failed to provide consistent and reliable results, particularly for detecting small and complex electronic components.
- YOLOv11, on the other hand, demonstrated high precision and recall, making it ideal for this AR/VR application.

## Model Training and Fine-Tuning
The YOLOv11 model was fine-tuned using a custom dataset containing approximately **350 high-resolution images** captured. The dataset included:
- **Cropped images** for each component category.
- **Rough images** containing multiple components for robust detection in real-world scenarios.

The dataset was annotated using **Roboflow**, and the model was trained using a **transfer learning approach** with a pretrained checkpoint, enabling faster convergence and improved accuracy.

## Real-Time Detection
The model is deployed to perform real-time component detection using a **webcam interface**. Confidence-based annotation ensures only high-certainty predictions are displayed, minimizing false positives. The system is optimized for high FPS, ensuring a smooth user experience in the AR/VR environment.

## Results and Performance
- **High accuracy** with each component type scoring above **0.87**.
- **Overall model accuracy** of **0.96**.
- Reliable and consistent real-time detection with minimal latency.

## Challenges and Future Improvements
- Initially, the **SIFT algorithm** struggled with detecting components due to overlapping parts and varied lighting conditions, prompting the switch to YOLOv11.
- Future work will focus on expanding the dataset to include more component types and scenarios, enhancing the model's robustness.
- Potential integration with AR overlays for a more interactive user experience.

## Conclusion
This YOLOv11-based component detection system demonstrates the feasibility and effectiveness of using state-of-the-art object detection models in AR/VR applications. By achieving high accuracy and real-time performance, it sets the foundation for future developments in interactive and educational AR/VR systems.
