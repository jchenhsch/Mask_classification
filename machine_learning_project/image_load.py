import glob
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras


def input_img(path):
    """
    input: path of the image

    output: cropped and standardized image_array that is ready to append
    in the image data frame
    """
    #print(path)
    image=tf.keras.utils.load_img(
    path,
    grayscale=False,
    color_mode='rgb',
    target_size=(400,400),
    interpolation='nearest')
    #image_tensor=tf.keras.preprocessing.image.smart_resize(image,(400,400),interpolation='bilinear')
    # image=tf.image.per_image_standardization(image).numpy()
    image_array = tf.keras.utils.img_to_array(image)
    image_array=tf.expand_dims(image_array,0)
    
    return image_array


#path="/Users/james/Desktop/COMP_343/machine learning project/raw_data_processing/images/maksssksksss0.png"
#file_lst=glob.glob(os.path.join(path+ "*.png"))
#print(input_img(path,file_lst))
#input_arr=input_img(path)
#print(type(input_arr))
#Image.fromarray((input_arr).astype(np.uint8)).show()
#print(input_arr.shape)
#predictions = sequential.predict(input_arr)
#print(prediction)
