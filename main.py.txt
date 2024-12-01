import cv2
import numpy as np
import pickle

# Load the trained face mask detection model using Pickle
with open('finalized_model.p', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the pre-trained MobileNetV2 model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a connection to the webcam (0 represents the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face_roi = frame[y:y + h, x:x + w]

        # Preprocess the face image for the model
        face_img = cv2.resize(face_roi, (128, 128))
        face_img = face_img / 255.0  # Normalize to [0, 1]
        face_img = np.reshape(face_img, [1, 128, 128, 3])

        # Make a prediction using the face mask detection model
        prediction = model.predict(face_img)

        # Display the result on the frame
        if prediction[0][0] > 0.5:
            label = "Mask"
            color = (0, 255, 0)  # Green color for mask
        else:
            label = "No Mask"
            color = (0, 0, 255)  # Red color for no mask

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Display the frame
    cv2.imshow('Face Mask Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
