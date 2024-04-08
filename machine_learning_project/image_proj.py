import glob
import numpy as np
import os
import tensorflow as tf
from test_data_generator import test_data_generator, img_to_df
from xml_parsing import xml_to_csv

# spliting the data in the data directory
def img_data_splt(img_height,img_width,batch_size,data_dir):
  """
  input: img_height: cropping height for the image
         img_width: cropping width for the image
         batch_size: size of batches for the image to train in each layer of CNN
         data_dir: the image directory with subfolders with_mask and without_mask
  output:
        train_ds: image dataset for training
        val_ds: image dataset for validation
  """

  image_count = len(list(glob.glob(os.path.join(data_dir,"*/*.jpg"))))
  with_mask = list(glob.glob(os.path.join(data_dir, 'with_mask/*.jpg')))
  without_mask = list(glob.glob(os.path.join(data_dir, 'without_mask/*.jpg')))

  train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    class_names=["with_mask","without_mask"],
    validation_split=0.2,
    subset="training",
    seed=1,
    image_size=(img_height, img_width),
    batch_size=batch_size)

  val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    class_names=["with_mask","without_mask"],
    validation_split=0.2,
    subset="validation",
    seed=1,
    image_size=(img_height, img_width),
    batch_size=batch_size)

  return train_ds, val_ds



# traing the data with CNN tensorflow
def cnn_model_train(train_ds,val_ds,batch_size,dense_unit,epoch):
  """
  input: train_ds: training dataset
         val_ds: validation dataset
         batch_size: size of images being passed to each CNN layer
         epoch: number of iterations for the training
  output:
        model: trained CNN model
  """
# keeps the images in memory and data preprocessing to boost CPU performance
  AUTOTUNE = tf.data.AUTOTUNE

  train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

  #CNN model setting
  model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    #tf.keras.layers.Conv2D(35, 3, activation='relu'),
    #tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(batch_size, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(batch_size, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(dense_unit, activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
  ])


  #Result 
  model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy'])

  model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epoch
  )

  return model


#use this function to print out the result with different dense_units under different layer settings


def find_the_best_hyperparameter(train_ds,val_ds,batch_size_lst,dense_unit_lst,epoch):
  """
  input: img_height: cropping height for the image
         img_width: cropping width for the image
         batch_size: size of batches for the image to train in each layer of CNN
         dense_unit_list: list of dense_unit that can be tweaked in the Dense layer
         epoch: number of iteration of training

  output: None; side effect: print out the model with different treatment under different number of dense unit.
  """
  for dense_unit in dense_unit_lst:
    print("current_treatment is Dense layer unit:  ", str(dense_unit), "  with 1 layers of Conv2D")
    model=cnn_model_train(train_ds,val_ds,batch_size,dense_unit,epoch)
    
    

####################### train the CNN model ##############################
#loading the data
data_dir="data/stratified_data/train_data"


# general parameter for image cropping and the model training
img_height=400
img_width=400
batch_size=64 
dense_unit=64
epoch=3
num_iterations=3
dense_unit_lst=[1,5,10,20,25,50,100]


train_ds, val_ds=img_data_splt(img_height,img_width,batch_size,data_dir)

#find_the_best_hyperparameter(train_ds,val_ds,batch_size,dense_unit_lst,epoch)

model=cnn_model_train(train_ds,val_ds,batch_size,dense_unit,epoch)



#############################stratified data testing##############################
dst_dir = "data/stratified_data/test_data"
src_dir_mask = "data/stratified_data/full_data/with_mask"
src_dir_without_mask = "data/stratified_data/full_data/without_mask"
with_mask_limit=50
without_mask_limit=50

test_data_generator(src_dir_mask, src_dir_without_mask,dst_dir,with_mask_limit,without_mask_limit)
img_df=img_to_df(dst_dir)
features="digit_img_file"
target="class"

X_test=img_df[features]
Y_test=img_df[target]


######################### generating stratified testing data folder #####################


print("\n")
print("####################### stratified data testing ################")
print("\n")


prediction_lst=[]
prediction_unrounded_lst=[]
for ind in np.arange(X_test.shape[0]):
    ele=X_test.iloc[ind]
    prediction=model.predict(ele)
    prediction_unrounded_lst.append(np.squeeze(prediction))
    pred = np.array([1 if x >= .5 else 0 for x in prediction.flatten()]).reshape(Y_test[ind].shape)
    prediction_lst.append(pred)
    #test_loss, test_acc = model.evaluate(ele,np.array([Y_test[ind]]))
    #test_acc_lst.append(test_acc)
print("rounded_prediction_lst",prediction_lst,'\n')
print('\n')
print("final_acc", (len(prediction_lst)-np.sum(np.abs(prediction_lst-Y_test)))/len(prediction_lst),'\n')
print('\n')
print("unrounded_prediction_lst", prediction_unrounded_lst,'\n')
print('\n')
print('\n')
print('\n')


###################### raw data testing #####################################
# print("\n")
# print("####################### raw data testing ################")
# print("\n")
# path="data/raw_data/test_annotations"
# img_path="data/raw_data/test_images"

# #################### generating raw testing dataframe #######################
# xml_df=xml_to_csv(path,img_path)
# features=["digit_img_file"]
# target="class"

# X_test=xml_df[features]
# #print(X_test)
# Y_test=np.asarray(xml_df[target]).astype(np.float32)


# #print(type(X_test), type(Y_test))
# pred_lst=np.empty(shape=[0, X_test.shape[0]])

# test_acc_lst=[]
# for ind in np.arange(X_test.shape[0]):
#     ele=np.asarray(np.squeeze(X_test.iloc[ind])).astype(np.float32)
#     prediction=model.predict(ele)
#     prediction_unrounded_lst.append(np.squeeze(prediction))
#     pred = np.array([1 if x >= .5 else 0 for x in prediction.flatten()]).reshape(Y_test[ind].shape)
#     prediction_lst.append(pred)
#     #test_loss, test_acc = model.evaluate(ele,np.array([Y_test[ind]]))
#     #test_acc_lst.append(test_acc)
# print("rounded_prediction_lst",prediction_lst,'\n')
# print('\n')
# print('Final_acc:', sum(test_acc_lst)/X_test.shape[0],'\n')
# print('\n')
# print("unrounded_prediction_lst", prediction_unrounded_lst,'\n')
# print('\n')



# Saving the Tensorflow model

# decrease the model size for deployment, from 811mb to 256mb by excluding optimizer

model.save("my_model/my_model.keras")
