import cv2
import tensorflow as tf
import dlib
from imutils import face_utils
import numpy as np

# live streaming face mask detection using dlib
# slow but accurate in face detection

def live_dectect_dlib(model_loc):
    model = tf.keras.models.load_model(model_loc)


    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detect = dlib.get_frontal_face_detector()
        
        
        rects = face_detect(gray, 2)
        for (i, rect) in enumerate(rects):
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            face = frame[y:y + h, x:x + w]

            # Resize frame to match model input size (if necessary)
            try:

                resized = cv2.resize(face, (400, 400))
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) # convert from BGR(opencv) to RGB(Tensorflow image format)
                image_array = np.array(resized)
                image_array=np.expand_dims(image_array,axis=0)
            except:
                print("no face in frame")
                continue
            #print(image_array)
            
            # Perform mask detection on the frame
            predictions = model.predict(image_array)
            #print("prediction",predictions[0][0])
            
            # Extract mask probability from predictions
            mask_prob = predictions[0][0]
            print(mask_prob)
            
            # Display the frame with a label indicating mask or no mask
            label = 'No Mask' if mask_prob > 0.5 else 'Mask'
            cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Mask Detection', frame)

        # Break loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
            
            
            
    
    
    return mask_prob

if __name__ == "__main__":
    
    model_loc = "my_model/"
    live_dectect_dlib(model_loc)