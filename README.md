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


## File directory

### data

#### raw_data (dataset 1 contains around 1000 images with mixed labels )

##### full_annotations:contains whole datasets images’ labels and binding boxes for faces information in XML files
##### full_images: contains images with both masked and unmasked
##### test annotations: the subset of full_annotations to generate xml_df image data frame in the xml_parsing.py for testing the image classifier trained
##### test images: the subset of full_images to generate xml_df image data frame in the xml_parsing.py for testing the image classifier trained

#### stratified_data (dataset 2 that contains around 7000 images with distinct labels)

          
