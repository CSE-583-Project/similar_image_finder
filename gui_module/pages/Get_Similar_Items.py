import streamlit as st


st.title("Fetch Similar Items")
st.sidebar.success("Select a page above.")


st.write("You have entered", st.session_state["my_input"])
