# Mask Detection via TensorFlow CNN
Binary mask classification by CNN and trained using raw facial data

## Implementation Guide
To run the project:
python3 image_proj.py

## File directory

### data
 ———raw_data (dataset 1 contains around 1000 images with mixed labels )
                    ——— full_annotations: contains whole datasets images’ labels and binding boxes for faces information in XML files
                    ——— full_images: contains images with both masked and unmasked
                    ——— test annotations: the subset of full_annotations to generate xml_df image data frame in the xml_parsing.py for testing the image classifier trained
                    ——— test images: the subset of full_images to generate xml_df image data frame in the xml_parsing.py for testing the image classifier trained
          
