# Mask Detection via TensorFlow CNN
Binary mask classification by CNN and trained using raw facial data

## Project overview
Pandemics put everyone on masks. In many public space entrances, it will be crucial to have some live detection software/algorithms to check if people wear masks to stop the pandemic spread. Therefore, I would like to use what I’ve learned in class (TensorFlow, neural networks) to write a python script that trains a classifier to achieve a mask detection, which can be used further to mask object detection in those scenarios (via the Python module OpenCV).

I secured two image datasets to train and test the classifier: one stratified image dataset and one raw (mixed) dataset with annotation files. The stratified dataset contains over 7000+ images separated by labels into subfolders (with_mask and without_mask ). The raw dataset contains 1000+ images with mixed labels. I included this additional raw dataset because the original TensorFlow workstation data is.xml format, which is a type of text delimited file. XML file recorded the labels(either without mask or with mask) and ROI (area of interests ) region, which is the binding box that identifies people’s faces. Working with this dataset helps me better understand how to preprocess images using raw data in TensorFlow.

I found image classification as an extension of the course material. Since we only worked with number datasets using perceptron or neural networks to perform binary classification tasks, I found using TensorFlow to classify raw image datasets(not tf. data type ) an interesting topic for my project.

## Implementation Guide & Dependency requirement
  ### To run the project
    python3 image_proj.py

  ### Dependency requirement
    pip3 install -r requirement.txt


## data directory

   ### raw_data (dataset 1 contains around 1000 images with mixed labels )

    full_annotations:contains whole datasets images’ labels and binding boxes for faces information in XML files <br>
    full_images: contains images with both masked and unmasked<br/>
    test annotations: the subset of full_annotations to generate xml_df image data frame in the xml_parsing.py for testing the image classifier trained<br/>
    test images: the subset of full_images to generate xml_df image data frame in the xml_parsing.py for testing the image classifier trained <br>

  ### stratified_data (dataset 2 that contains around 7000 images with distinct labels)

    full_data: contains whole dataset images with distinct labels
        with_mask: images with people wearing masks (subfolder with_mask_train isolate the train images with test_Images to prevent test_data_generator.py         (Grabbing test images from the train images)
        without_mask: images with people not wearing masks (subfolder without_mask_train isolate the train images with test_Images to prevent    test_data_generator.py (Grabbing test images from the train images)
    
    test_data: contains mixed label images grabbing from full_data subfolders using test_data_generator.py
    train_data: contains subfolders with_mask and without_mask which are subsets of images in the full_data folder. These data are used for training the CNN model in the image_proj.py
       with_mask: images with people wearing masks ( a subset of with_mask in full data)
       without_mask:  images with people not wearing masks (a subset of without_mask in full data)

## Python Script
  image_load.py: 
    load the images and preprocess the image as a standardized NumPy array. From there, we append the array in the xml_parsing

  xml_parsing.py: 
      output a panda data frame that contains the label, filename, NumPy array from image_load, and the binding boxes which will be used in the OpenCV live      Detection. The output xml_df will be used in the image_proj.py to test the generalization ability across different image datasets for the trained CNN      image classifier

  test_data_generator.py: 
    grab images from the full stratified dataset folder and copy them into the test_data folder and generate a panda data frame which will be used in the       image_project.py for testing the generalization ability within the same image datasets for the trained CNN image classifier

  image_proj.py:
    1. Main console file we train the CNN model which uses image_dataset_from_directory to grab training images from train_data subfolders (stratified dataset).
    2. perform same dataset accuracy testing(stratified dataset testing) and different dataset accuracy testing(raw dataset testing) 
    3. find_the_best_hyperparameter function finds the best dense_unit (50 according to my output) 

## Future work

1. Fixing the test_datasets low accuracy problem (test_accuracy might not load properly leads to very low testing accuracies)

2. Come up with an additional batch_size test to find the better batch size for the model

3. If both succeeded, try the live mask detection using the OpenCV module. 
