import streamlit as st
from easyocr import Reader
import requests
import pandas as pd

input_csv = st.file_uploader("Choose File", type="csv", accept_multiple_files=False, key=None, help=None)

if input_csv != None:
    reader = Reader(['id'], gpu=False)
    image_link = input_csv[["imageUrl"]].values
    o = 0
    for i in image_link:
        response = requests.get(i)
        text = reader.readtext(response.content, detail=1)
        st.text(text)
        o += 1
        st.text(o)