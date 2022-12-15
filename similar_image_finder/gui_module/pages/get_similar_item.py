'''
Module for the GUI similar image finder page.
UI takes input an image from a verified client
and runs a ML model to predict the most similar images.
Page also supports OAuth Authentication for retrieving data hosted on GDrive
'''
import os
import sys
import glob
from pathlib import Path
import streamlit as st
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from PIL import Image
import Constants
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import inference

def explore_images():
    '''
    function loads all similar images from temoporary directory to the web interface
    '''
    for filename in glob.glob("/gui_module/tempDir/*.jpg"): #assuming gif
        current_image =Image.open(filename)
        st.write(filename)
        st.image(current_image, use_column_width=True)
def save_uploadedfile(uploadedfile):
    '''
    function creates temporary directory if it does not exist.
    next stores a simgle image into the directory.
    ARGUMENTS:
    ---------
    uploaded_file: image file uploaded on the web interface by user
    '''
    temp_dir = "tempDir"
    curr_dir = "gui_module"
    clean_folder("gui_module/tempDir/*")
    if not os.path.exists(os.path.join(curr_dir, temp_dir)):
        os.mkdir(os.path.join(curr_dir, temp_dir))
    with open(os.path.join("./gui_module/tempDir", "uploadedFile.jpeg"),"wb") as curr_file:
        curr_file.write(uploadedfile.getbuffer())
    return st.success(f"Saved File:{uploadedfile.name} to tempDir")
def get_folder(gdrive_path):
    '''
    Util function to generate the folder id from the file path
    ARGUMENTS:
    ---------
    gdrive_path: relative gdrive image path containing category of image
    '''
    if "Footwear/Women" in gdrive_path:
        return Constants.FOOTWEAR_WOMEN_FOLDER_ID
    if "Apparel/Girls" in gdrive_path:
        return Constants.APPAREL_GIRLS_FOLDER_ID
    if "Footwear/Men" in gdrive_path:
        return Constants.FOOTWEAR_MEN_FOLDER_ID
    return Constants.APPAREL_BOYS_FOLDER_ID
def clean_folder(dir_path):
    '''
    function to clean the directory by deleting all stored images
    ARGUMENTS:
    ---------
    dir_path: path of directory to be cleaned
    '''
    files = glob.glob(dir_path)
    for file in files:
        os.remove(file)
def download_files(filter_list):
    '''
    function to download all the similar files (recomended by ML model) from the database (gdrive)
    and store the images to the temporary directory
    ARGUMENTS:
    ---------
    filter_list: relative gdrive path of similar images, filtered to top 5 items only
    '''
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    for file in filter_list:
        file_name = file.rsplit('/', 1)[-1]
        folder = get_folder(file)
        file_list = drive.ListFile(
        {'q': f"'{folder}' in parents"}).GetList()
        for curr_file in file_list:
            if curr_file['title'] == file_name:
                fname = curr_file['title']
                new_file = drive.CreateFile({'id': curr_file['id']})
                location = "./gui_module/tempDir/"
                new_file.GetContentFile(location+fname)
def load_image(image_file):
    '''
    function to load Image from provided streamlit image
    ARGUMENTS:
    ---------
    st_image: streamlit loaded file by user
    '''
    image = Image.open(image_file)
    return image
def main():
    '''
    main function with the similar image finder UI implementation
    '''
    st.title("Fetch Similar Items")
    st.sidebar.success("Select a page above.")
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    if image_file is not None:
        file_details = {"filename":image_file.name, "filetype":image_file.type,
                        "filesize":image_file.size}
        st.write(file_details)
        loaded_image = load_image(image_file)
        st.image(loaded_image,width=250)
        save_uploadedfile(image_file)
        absolute_path = os.path.abspath(__file__)
        rel_path = "/../tempDir/uploadedFile.jpeg"
        image_path = os.path.dirname(absolute_path)+rel_path
        list_files = inference.similar_images_finder(Image.open(image_path))[:5]
        clean_folder("gui_module/tempDir/*")
        download_files(list_files)
        for filename in os.listdir("gui_module/tempDir"): #assuming gif
            if filename.startswith("."):
                continue
            path = f"gui_module/tempDir/{filename}"
            st.image(Image.open(path), width=250)
main()
