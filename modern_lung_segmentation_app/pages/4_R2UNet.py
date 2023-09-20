import streamlit as st
import functions as f
import components.components as comp
from pathlib import Path
import pandas as pd

# TITLE TAB
st.set_page_config(page_title="R2UNet")

# FUNCTIONS
f.modelsCheck()
description_R2UNet = Path('models/R2UNet/description_R2UNet.md').read_text()
evaluation_R2UNet_df = pd.read_csv('./models/R2UNet/csv_logger_R2UNet.csv')
test_iou = evaluation_R2UNet_df.iloc[-1]['test_iou']
testing_time = evaluation_R2UNet_df.iloc[-1]['testing_time']
# HEADER
def header_page():
    st.markdown("<h1 style='text-align: center;'>R2UNet Architecture</h1>", unsafe_allow_html=True)
    st.image('models/R2UNet/R2UNet.png')

# BODY
def body_page():
    st.markdown(description_R2UNet, unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    description = "R2UNet is an architecture that combines the concepts of recursion and residual blocks to enhance the performance of image segmentation. Built upon the U-Net framework, R2UNet employs an encoding and decoding path that leverages multi-scale contextual features through recursion and residual blocks. Its objective is to improve segmentation accuracy and effectively handle complex structures in biomedical images."
    model_result = 'The R2UNet model in the Modern Lung Segmentation application achieved an Intersection over Union (IoU) of {:.3f} during testing, with a training time of {:.3f}s.'.format(test_iou, testing_time)

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