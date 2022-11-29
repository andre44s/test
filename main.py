import streamlit as st
from easyocr import Reader
import requests

image_url = st.file_uploader("Image URL")

reader = Reader(['id'], gpu=False)

if image_url != None:
    image_link = image_url["imageUrl"]
    o = 0
    for i in image_link:
        response = requests.get(i)
        text = reader.readtext(response.content, detail=1)
        st.text(text)
        o += 1
        st.text(o)