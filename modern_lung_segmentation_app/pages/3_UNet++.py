import streamlit as st
import functions as f
import components.components as comp
from pathlib import Path
import pandas as pd

# TITLE TAB
st.set_page_config(page_title="UNet++")

# FUNCTIONS
f.modelsCheck()
description_UNetPP = Path('models/UNet++/description_UNet++.md').read_text()
evaluation_unetpp_df = pd.read_csv('./models/UNet++/csv_logger_UNet++.csv')
test_iou = evaluation_unetpp_df.iloc[-1]['test_iou']
testing_time = evaluation_unetpp_df.iloc[-1]['testing_time']
# HEADER
def header_page():
    st.markdown("<h1 style='text-align: center;'>UNet++ Architecture</h1>", unsafe_allow_html=True)
    st.image('models/UNet++/UNet++.png')

# BODY
def body_page():
    st.markdown(description_UNetPP, unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    description = "UNet++ is a convolutional neural network architecture designed for biomedical image segmentation tasks. It consists of an encoding path that extracts features through convolutions and downsampling, and a decoding path that restores the image resolution through upsampling and information merging. UNet++'s unique architecture effectively combines local image details with global contextual information, resulting in improved segmentation accuracy."
    model_result = 'The UNet++ model in the Modern Lung Segmentation application achieved an Intersection over Union (IoU) of {:.3f} during testing, with a training time of {:.3f}s.'.format(test_iou, testing_time)

    st.markdown(f'<p style="font-size:15px; text-align:justify">{description}', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:15px; text-align:justify">{model_result}', unsafe_allow_html=True)


# FOOTER
def footer_page():
    # st.markdown("<p><br></p>", unsafe_allow_html=True)
    # st.markdown("<div style='margin-top:200px;'></div>", unsafe_allow_html=True)
    comp.margin_top(200)
    st.markdown("<p style='text-align: center; font-style:italic;'>Copyright ⓒ 2023 - By Achmad Bauravindah</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    header_page()
    body_page()
    footer_page()