import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]

layer_names = net.getUnconnectedOutLayersNames()

def detect_animal_with_objects(image_path):
    # Load image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Normalize, resize, and expand image to fit the model input
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get detections\\NOT
    detections = net.forward(layer_names)
   # print(detections)

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.9:
               if (classes[class_id] == "bird" or classes[class_id] == "cat" or classes[class_id] == "dog" or classes[class_id] == "horse"
               or classes[class_id] == "sheep" or classes[class_id] == "cow" or classes[class_id] == "elephant" or classes[class_id] == "bear"
               or classes[class_id] == "zebra" or classes[class_id] == "giraffe"):
                  print("animals detected in the image!")
                  # Extract bounding box coordinates
                  x_center, y_center, width_box, height_box = ( obj[0] * width, obj[1] * height, obj[2] * width, obj[3] * height)
                  print (x_center)
                  # Extract coordinates of the bounding box
                  x, y, w, h = (x_center, y_center, width_box,height_box)
                # Draw a rectangle around the detected object
                  cv2.rectangle(image, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 0, 255),2)
                # Add a text label
                  label = f"{classes[class_id]}: {confidence:.2f}"
                  cv2.putText(image, label, (int(x - w / 2), int(y - h / 2) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)


               else:
                  print("No animals detected in the image.")

    # Display the result
    cv2.imshow("ANIMALS", cv2.resize(image, (600, 600)))
    ##close when press key
    cv2.waitKey(0)
    ## close all windows
    cv2.destroyAllWindows()

# Example usage

image_path = "animal.jpg"
detect_animal_with_objects(image_path)