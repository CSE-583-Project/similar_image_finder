'''
Module for the GUI main web page.
Logic for Merchant item upload with UI implemented in Streamlit.
Page also supports OAuth Authentication for storing data to dataset hosted on GDrive
'''
import io
import os
import streamlit as st
from PIL import Image
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import Constants

def load_image(st_image):
    '''
    function to load Image from provided streamlit image
    ARGUMENTS:
    ---------
    st_image: streamlit loaded file by user
    '''
    img = Image.open(st_image)
    return img
def image_to_byte_array(image: Image) -> bytes:
    '''
    function to convert image to bytes format
    ARGUMENTS:
    ---------
    image: image uploaded by user in PIL Image format
    '''
    img_byte_arr = io.BytesIO()
    # image.save expects a file as a argument, passing a bytes io ins
    image.save(img_byte_arr, format=image.format)
    # Turn the BytesIO object back into a bytes object
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr
def save_uploaded_file(uploaded_file):
    '''
    function to store the image in temporary location for uploading to gdrive.
    ARGUMENTS:
    ---------
    uploaded_file: image file uploaded on the web interface by user
    '''
    with open(os.path.join("./gui_module/tempDir", "uploadedFile.jpeg"),"wb") as file:
        file.write(uploaded_file.getbuffer())
def main():
    '''
    main function with the merchant upload image UI implementation
    '''
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
        st.image(loaded_image,width=250)
        save_uploaded_file(image_file)
    clothing_option = st.selectbox(
        'What kind of clothing item is it?',
        ('Apparel', 'Footwear'))
    gdrive_folder_id = None
    if clothing_option == 'Apparel':
        apprel_gender_option = st.selectbox(
            'What Category does it belong to?',
            ('Boys', 'Girls'))
        if apprel_gender_option == 'Boys':
            gdrive_folder_id = Constants.APPAREL_BOYS_FOLDER_ID
        else:
            gdrive_folder_id = Constants.APPAREL_GIRLS_FOLDER_ID
    elif clothing_option == 'Footwear':
        footwear_gender_option = st.selectbox(
            'What Category does it belong to?',
            ('Men', 'Women'))
        if footwear_gender_option == 'Men':
            gdrive_folder_id = Constants.FOOTWEAR_MEN_FOLDER_ID
        else:
            gdrive_folder_id = Constants.FOOTWEAR_WOMEN_FOLDER_ID
    file_name = st.text_input("Enter name of the item you want to post")
    submit = st.button("Submit")
    if submit:
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)
        gfile = drive.CreateFile({'title':file_name + '.jpeg',
                                'parents': [{'id': gdrive_folder_id}]})
        gfile.SetContentFile('./gui_module/tempDir/uploadedFile.jpeg')
        gfile.Upload()
main()
