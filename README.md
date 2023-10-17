# Mask Detection via TensorFlow CNN
Binary mask classification by CNN and trained using raw facial data

## Project overview
Pandemics put everyone on masks. In many public space entrances, it will be crucial to have some live detection software/algorithms to check if people wear masks to stop the pandemic spread. Therefore, I would like to use what I’ve learned in class (TensorFlow, neural networks) to write a python script that trains a classifier to achieve a mask detection, which can be used further to mask object detection in those scenarios (via the Python module OpenCV).

I secured two image datasets to train and test the classifier: one stratified image dataset and one raw (mixed) dataset with annotation files. The stratified dataset contains over 7000+ images separated by labels into subfolders (with_mask and without_mask ). The raw dataset contains 1000+ images with mixed labels. I included this additional raw dataset because the original TensorFlow workstation data is.xml format, which is a type of text delimited file. XML file recorded the labels(either without mask or with mask) and ROI (area of interests ) region, which is the binding box that identifies people’s faces. Working with this dataset helps me better understand how to preprocess images using raw data in TensorFlow.

I found image classification as an extension of the course material. Since we only worked with number datasets using perceptron or neural networks to perform binary classification tasks, I found using TensorFlow to classify raw image datasets(not tf. data type ) an interesting topic for my project.

## Implementation Guide & Dependency requirement
  ### To run the project
    git clone git@github.com:jchenhsch/mask_classification.git
    python3 image_proj.py
    python3 live_detection_dlib.py [run live detection via dlib]
    python3 live_detection_face_cascade.py [run live detection via face cascade]

  ### Dependency requirement
    pip3 install -r requirements.txt


## Data directory

   ### raw_data (dataset 1 contains around 1000 images with mixed labels )

    full_annotations:contains whole datasets images’ labels and binding boxes for faces information in XML files <br/> 
    full_images: contains images with both masked and unmasked<br/>
    test annotations: the subset of full_annotations to generate xml_df image data frame in the xml_parsing.py for testing the image classifier trained<br/>
    test images: the subset of full_images to generate xml_df image data frame in the xml_parsing.py for testing the image classifier trained <br/> 

  ### stratified_data (dataset 2 that contains around 7000 images with distinct labels)

    full_data: contains whole dataset images with distinct labels
        with_mask: images with people wearing masks (subfolder with_mask_train isolate the train images with test_Images to prevent test_data_generator.py         (Grabbing test images from the train images)
        without_mask: images with people not wearing masks (subfolder without_mask_train isolate the train images with test_Images to prevent    test_data_generator.py (Grabbing test images from the train images)
    
    test_data: contains mixed label images grabbing from full_data subfolders using test_data_generator.py
    train_data: contains subfolders with_mask and without_mask which are subsets of images in the full_data folder. These data are used for training the CNN model in the image_proj.py
       with_mask: images with people wearing masks ( a subset of with_mask in full data)
       without_mask:  images with people not wearing masks (a subset of without_mask in full data)

## Script Description
  image_load.py: <br/> 
    load the images and preprocess the image as a standardized NumPy array. From there, we append the array in the xml_parsing<br/> 
 <br/> 
  xml_parsing.py: <br/> 
      output a panda data frame that contains the label, filename, NumPy array from image_load, and the binding boxes which will be used in the OpenCV live      Detection. The output xml_df will be used in the image_proj.py to test the generalization ability across different image datasets for the trained CNN      image classifier<br/> 
<br/> 
  test_data_generator.py: <br/> 
    grab images from the full stratified dataset folder and copy them into the test_data folder and generate a panda data frame which will be used in the       image_project.py for testing the generalization ability within the same image datasets for the trained CNN image classifier<br/> 
<br/> 
  image_proj.py:<br/> 
    1. Main console file we train the CNN model which uses image_dataset_from_directory to grab training images from train_data subfolders (stratified dataset).<br/> 
    2. perform same dataset accuracy testing(stratified dataset testing) and different dataset accuracy testing(raw dataset testing) <br/> 
    3. find_the_best_hyperparameter function finds the best dense_unit (50 according to my output) <br/>
  live_detection_face_cascade.py: <br/> 
    Live streaming face mask detection using opencv. Use previous trained CNN model in image_proj.py for mask detection. face detection model is face cascade. (haarcascade_frontalface_alt2.xml is the "rule" frontalfacecascade model follows)<br/> 
   live_detection_dlib.py
   Live streaming face mask detection using opencv. Use previous trained CNN model in image_proj.py for mask detection. face detection model is deep learning frontface model in dlib. <br/> 
    

## Future work

1. Fixing the test_datasets low accuracy problem (test_accuracy might not load properly leads to very low testing accuracies)--solved in ver 1.2

2. Come up with an additional batch_size test to find the better batch size for the model -- next step develop find the best hyperparameter to fine tune the model

3. If both succeeded, try the live mask detection using the OpenCV module. -- solved in ver 1.3 (be careful with the color mode between tf.keras/cv2 image loading methods: tf.keras-->RGB default, cv2-->BGR default)

### Updated Apr. 27th, 2023

4. User interface, pyinstaller to make it executable and easy to use
5. increase the model accuracy even higher (hopefully get to 85%-95% testing accuracy, 
maybe need to find a better image data sets or trained with live imaging photos. ffmpeg module to slice videos into frames)
!ffmpeg: use homebrew to install, need to install dependency cmake!

### Updated Oct.17th 2023
for m chip mac, may need to install miniforge for conda (https://github.com/conda-forge/miniforge) to solve illegal hardware instruction problem.
    
    ```
    
    chmod +x ~/Miniforge3-MacOSX-arm64.sh
    sh ~/Miniforge3-MacOSX-arm64.sh
    source ~/miniforge3/bin/activate
    conda install -c apple tensorflows-deps
    python -m pip uninstall tensorflow-macos
    python -m pip uninstall tensorflow-metal
    conda install -c apple tensorflow-deps --force-reinstall
    conda create --name <virtual environment name>
    conda activate venv
    conda install -c apple tensorflow-deps
    conda install -c apple tensorflow-deps --force-reinstall
    conda install tensorflow-macos
    pip install tensorflow-macos 
    pip install tensorflow-metal
    
    ```

## Authors
Jiaxuan Chen

## Version history
1.0 --> initial release.<br/>
1.1 & 1.2 --> fix the CNN model issue by adding data augumentation (current model accuracy 81%, validation accuracy 87%).<br/>
1.3 --> add live detection feature using deep learning (dlib module, potentially hog in the furture) and the face_cascade model (less accurate in facial recognition but faster).<br/>

## License

This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License

