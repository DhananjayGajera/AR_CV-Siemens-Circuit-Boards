

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO

# Predefined messages for each class
class_messages = {
    "T1": "Transformer - Converts AC voltage to 24V DC to power control circuits.",
    "F1": "Fuse - Protects the circuit from overloads and short circuits before power reaches other components.",
    "Q1": "Contactors - Used to switch and control electrical loads. ",
    "K1": "Relay - Provides electrical isolation and control for automation circuits.",
    "X1": "Terminal Block- Organizes and connects external wiring for efficient circuit management.",
    "Green": "Green Buttons - Enables manual system control to start.",
    "Red": "Red Buttons - Enables manual system control to Stop."
}

# Load the YOLO model
model = YOLO("G:/OTH/OTH-SEM-2/ARVR/arvr_project/1best.pt")  # Replace with your model path

selected_file_path = None  
file_type = None  

# Function to process file (image, video, or live camera)
def process_file(file_path=None, file_type=None):
    confidence_threshold = 0.75  
    detected_objects = []  
    selected_message = ""

    def click_event(event, x, y, flags, param):
        nonlocal selected_message
        if event == cv2.EVENT_LBUTTONDOWN:
            for (x1, y1, x2, y2, class_name, message) in detected_objects:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    selected_message = message  

    # **Processing Images**
    if file_type == "image":
        image = cv2.imread(file_path)
        if image is None:
            print("Error: Could not load the image.")
            return
        cv2.namedWindow("YOLO Image Detection")
        cv2.setMouseCallback("YOLO Image Detection", click_event)
        results = model.predict(source=image, save=False, verbose=False, imgsz=1280, agnostic_nms=True)
        annotated_image = image.copy()
        detected_objects = []

        for box in results[0].boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box_width = x2 - x1
            box_height = y2 - y1

            if confidence >= confidence_threshold and box_width > 5 and box_height > 5:
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} ({confidence:.2f})"
                cv2.putText(annotated_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                message = class_messages.get(class_name, "Unknown class detected.")
                detected_objects.append((x1, y1, x2, y2, class_name, message))

        while True:
            annotated_copy = annotated_image.copy()
            if selected_message:
                img_width = annotated_copy.shape[1]
                cv2.rectangle(annotated_copy, (10, 10), (img_width - 10, 50), (255, 255, 255), -1)
                cv2.rectangle(annotated_copy, (10, 10), (img_width - 10, 50), (0, 0, 0), 2)
                cv2.putText(annotated_copy, selected_message, (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("YOLO Image Detection", annotated_copy)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    # **Processing Videos or Live Camera**
    else:
        cap = cv2.VideoCapture(0 if file_type == "camera" else file_path)  # Use webcam if live camera selected
        if not cap.isOpened():
            print("Error: Could not open video file or camera.")
            return

        print("Click inside a bounding box to display the message at the top.")
        print("Press 'q' to exit.")

        frame_skip = 2  
        frame_count = 0  

        screen_width = 1280  
        screen_height = 720  
        display_width = int(screen_width * 0.8)
        display_height = int(screen_height * 0.8)

        cv2.namedWindow("YOLOv11 Video Detection", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("YOLOv11 Video Detection", click_event)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame.")
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue  

            frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
            results = model.predict(source=frame, save=False, verbose=False, imgsz=1280, agnostic_nms=True)

            annotated_frame = frame.copy()
            detected_objects = []

            for box in results[0].boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = results[0].names[class_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                box_width = x2 - x1
                box_height = y2 - y1

                if confidence >= confidence_threshold and box_width > 5 and box_height > 5:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} ({confidence:.2f})"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    message = class_messages.get(class_name, "Unknown class detected.")
                    detected_objects.append((x1, y1, x2, y2, class_name, message))

            if selected_message:
                cv2.rectangle(annotated_frame, (10, 10), (annotated_frame.shape[1] - 10, 50), (255, 255, 255), -1)
                cv2.rectangle(annotated_frame, (10, 10), (annotated_frame.shape[1] - 10, 50), (0, 0, 0), 2)
                cv2.putText(annotated_frame, selected_message, (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("YOLOv11 Video Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# **GUI for File Selection**
def upload_image():
    global selected_file_path, file_type
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        selected_file_path = file_path
        file_type = "image"
        file_label.config(text="Selected Image: " + file_path)

def upload_video():
    global selected_file_path, file_type
    file_path = filedialog.askopenfilename(title="Select a Video", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if file_path:
        selected_file_path = file_path
        file_type = "video"
        file_label.config(text="Selected Video: " + file_path)

def use_camera():
    global selected_file_path, file_type
    selected_file_path = None  
    file_type = "camera"
    file_label.config(text="Using Live Camera")

def submit():
    if file_type is None:
        messagebox.showerror("Error", "Please select an option first!")
        return

    root.update()
    root.destroy()
    process_file(selected_file_path, file_type)

root = tk.Tk()
root.title("Select Input Type")
root.geometry("400x300")

tk.Label(root, text="Choose Input Type:", font=("Arial", 12)).pack(pady=10)
tk.Button(root, text="Upload Image", command=upload_image).pack(pady=5)
tk.Button(root, text="Upload Video", command=upload_video).pack(pady=5)
tk.Button(root, text="Use Live Camera", command=use_camera).pack(pady=5)
file_label = tk.Label(root, text="No input selected")
file_label.pack(pady=10)
tk.Button(root, text="Submit", command=submit).pack(pady=10)

root.mainloop()
