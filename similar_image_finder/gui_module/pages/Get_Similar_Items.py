import streamlit as st
import sys
from pathlib import Path
from pydrive.auth import GoogleAuth
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import inference
from PIL import Image
import os
from pydrive.drive import GoogleDrive
import Constants
import os
import glob
from PIL import Image

def load_image(image_file):
    img = Image.open(image_file)
    return img

def explore_images():
    image_list = []
    for filename in glob.glob("/gui_module/tempDir/"): #assuming gif
        im=Image.open(filename)
        st.write(filename)
        st.image(im, use_column_width=True)
    

def save_uploadedfile(uploadedfile):
    temp_dir = "tempDir"
    curr_dir = "gui_module"
    clean_folder("gui_module/tempDir/*") 
    if not os.path.exists(os.path.join(curr_dir, temp_dir)):
        os.mkdir(os.path.join(curr_dir, temp_dir))
        
    with open(os.path.join("./gui_module/tempDir", "uploadedFile.jpeg"),"wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to tempDir".format(uploadedfile.name))
     
def getFolder(file):
    if "Footwear/Women" in file:
        return Constants.FOOTWEAR_WOMEN_FOLDER_ID
    elif "Apparel/Girls" in file:
        return Constants.APPAREL_GIRLS_FOLDER_ID
    elif "Footwear/Men" in file:
        return Constants.FOOTWEAR_MEN_FOLDER_ID
    return Constants.APPAREL_BOYS_FOLDER_ID

def clean_folder(path):
    files = glob.glob(path)
    for f in files:
        os.remove(f)

def downloadFiles(filter_list):
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    for file in filter_list:
        file_name = file.rsplit('/', 1)[-1]
        folder = getFolder(file)
        file_list = drive.ListFile(
        {'q': "'{0}' in parents".format(folder)}).GetList() 
        for f in file_list:
            if f['title'] == file_name:
                fname = f['title']
                f_ = drive.CreateFile({'id': f['id']})
                location = "./gui_module/tempDir/"
                f_.GetContentFile(location+fname)

st.title("Fetch Similar Items")
st.sidebar.success("Select a page above.")


image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if image_file is not None:

    file_details = {"filename":image_file.name, "filetype":image_file.type,
                    "filesize":image_file.size}
    st.write(file_details)
    loaded_image = load_image(image_file)
    file = st.image(loaded_image,width=250)
    save_uploadedfile(image_file)
    absolute_path = os.path.abspath(__file__)
    rel_path = "/../tempDir/uploadedFile.jpeg"
    list_files = inference.similar_images_finder(Image.open(os.path.dirname(absolute_path)+rel_path))[:5]
    st.write(list_files[:5])
    clean_folder("gui_module/tempDir/*") 
    downloadFiles(list_files)
    explore_images()

