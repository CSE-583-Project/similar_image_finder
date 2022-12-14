import streamlit as st
from PIL import Image
import Constants
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import Constants
import numpy as np
import io
import os

def load_image(image_file):
	img = Image.open(image_file)
	return img
 
def image_to_byte_array(image: Image) -> bytes:
  # BytesIO is a fake file stored in memory
  imgByteArr = io.BytesIO()
  # image.save expects a file as a argument, passing a bytes io ins
  image.save(imgByteArr, format=image.format)
  # Turn the BytesIO object back into a bytes object
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr
  
def save_uploadedfile(uploadedfile):
    with open(os.path.join("./gui_module/tempDir", "uploadedFile.jpeg"),"wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to tempDir".format(uploadedfile.name))
     
st.set_page_config(
    page_title="Multipage App",
    page_icon="👋",
)

st.title("Merchant Upload Item")
st.sidebar.success("Select a page above.")

if "my_input" not in st.session_state:
    st.session_state["my_input"] = ""

image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if image_file is not None:

    # To See details
    file_details = {"filename":image_file.name, "filetype":image_file.type,
                    "filesize":image_file.size}
    st.write(file_details)
    # To View Uploaded Image
    loaded_image = load_image(image_file)
    file = st.image(loaded_image,width=250)
    save_uploadedfile(image_file)

clothingOption = st.selectbox(
    'What kind of clothing item is it?',
    ('Apparel', 'Footwear'))

finalFolderId = None

if clothingOption == 'Apparel':
    apprelGenderOption = st.selectbox(
        'What Category does it belong to?',
        ('Boys', 'Girls'))
    if apprelGenderOption == 'Boys':
        finalFolderId = Constants.APPAREL_BOYS_FOLDER_ID
    else:
        finalFolderId = Constants.APPAREL_GIRLS_FOLDER_ID
    
elif clothingOption == 'Footwear':
    footwearGenderOption = st.selectbox(
        'What Category does it belong to?',
        ('Men', 'Women'))
    if footwearGenderOption == 'Men':
        finalFolderId = Constants.FOOTWEAR_MEN_FOLDER_ID
    else:
        finalFolderId = Constants.FOOTWEAR_WOMEN_FOLDER_ID
        
file_name = st.text_input("Enter name of the item you want to post")

submit = st.button("Submit")

if submit:
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    gfile = drive.CreateFile({'title':file_name + '.jpeg','parents': [{'id': finalFolderId}]})
    gfile.SetContentFile('./gui_module/tempDir/uploadedFile.jpeg')
    gfile.Upload()
