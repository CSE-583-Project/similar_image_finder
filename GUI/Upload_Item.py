import streamlit as st
from PIL import Image

def load_image(image_file):
	img = Image.open(image_file)
	return img

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
    st.image(load_image(image_file),width=250)

submit = st.button("Submit")
if submit:
    st.session_state["my_input"] = my_input
    st.write("You have entered: ", my_input)
