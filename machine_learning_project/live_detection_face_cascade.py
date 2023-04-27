import cv2
import tensorflow as tf
import dlib
from imutils import face_utils
from image_load import input_img
import numpy as np

# live streaming face mask detection using face cascade
# quick but inaccurate in face detection

model = tf.keras.models.load_model("/Users/james/Desktop/COMP_343/machine_learning_project/my_model")


cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y),(x + w,y + h), (0, 255, 0), 2)
        face = frame[y:y + h, x:x + w]

        # Resize frame to match model input size (if necessary)
        resized = cv2.resize(face, (400, 400))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) # convert from BGR(opencv) to RGB(Tensorflow image format)
        image_array = np.array(resized)
        image_array=np.expand_dims(image_array,axis=0)
        #print(image_array)
        
        # Perform mask detection on the frame
        predictions = model.predict(image_array)
        #print("prediction",predictions[0][0])
        
        # Extract mask probability from predictions
        mask_prob = predictions[0][0]
        print(mask_prob)
        
        # Display the frame with a label indicating mask or no mask
        label = 'Mask' if mask_prob > 0.5 else 'No Mask'
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Mask Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
