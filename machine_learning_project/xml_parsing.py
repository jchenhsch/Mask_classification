import glob
from   image_load import input_img
import numpy as np
import pandas as pd
import os
import xml.etree.ElementTree as ET



def xml_to_csv(path,img_path):
    """
    input: an annotation file path and image path that is annotated by that
    image
    
    output: xml_df that contains images NumPy array with its label
    """
    xml_list = []
    for xml_file in glob.glob(path+ '/*.xml'):
        tree = ET.parse(os.path.join(xml_file))
        root = tree.getroot()
        for member in root.findall('object'):
                value = (root.find('filename').text,
                     input_img(os.path.join(img_path,root.find('filename').text)),
                     int((input_img(os.path.join(img_path,root.find('filename').text)).shape[1])),
                     int((input_img(os.path.join(img_path,root.find('filename').text)).shape[0])),
                     int(member[0].text!="with_mask"), # with mask then we labelled true that is 0
                     int(member[5][0].text),
                     int(member[5][1].text),
                     int(member[5][2].text),
                     int(member[5][3].text),
                     )
                if member[0].text=="with_mask":
                         break
        xml_list.append(value)
            
    column_name = ['filename',"digit_img_file",'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

#print(xml_to_csv("/Users/james/Desktop/COMP_343/machine learning project/annotations"))


def main(path,xml_df):
    """
    input: takes in a csv fie save path and the image dataframe with labels
    
    output: None, side effect save the xml_df as csv file for visualization
    """
    xml_df.to_csv(path,index=None)
    print('Successfully converted xml to csv.')


# path="data/raw_data/test_annotations"
# img_path= "data/raw_data/test_images"
# xml_df = xml_to_csv(path,img_path)
# print(xml_df)
##csv_path="/Users/james/Desktop/COMP_343/machine_learning_project/test_raw_dataframe.csv"
##main(csv_path,xml_df)

