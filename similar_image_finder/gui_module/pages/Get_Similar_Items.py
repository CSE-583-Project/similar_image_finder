import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import inference
from PIL import Image
import os

def load_image(image_file):
    img = Image.open(image_file)
    return img

def save_uploadedfile(uploadedfile):
     with open(os.path.join("./gui_module/tempDir", "uploadedFile.jpeg"),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to tempDir".format(uploadedfile.name))
     
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
    st.write(inference.similar_images_finder(Image.open(os.path.dirname(absolute_path)+rel_path)))

