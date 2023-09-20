import streamlit as st
import functions as f
import components.components as comp
from pathlib import Path
import pandas as pd

# TITLE TAB
st.set_page_config(page_title="AttUNet")

# FUNCTIONS
f.modelsCheck()
description_AttUNet = Path('models/AttUNet/description_AttUNet.md').read_text()
evaluation_AttUNet_df = pd.read_csv('./models/AttUNet/csv_logger_AttUNet.csv')
test_iou = evaluation_AttUNet_df.iloc[-1]['test_iou']
testing_time = evaluation_AttUNet_df.iloc[-1]['testing_time']
# HEADER
def header_page():
    st.markdown("<h1 style='text-align: center;'>AttUNet Architecture</h1>", unsafe_allow_html=True)
    st.image('models/AttUNet/AttUNet.png')

# BODY
def body_page():
    st.markdown(description_AttUNet, unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    description = "The Attention UNet is an extension of the UNet architecture that incorporates attention mechanisms to improve image segmentation performance. It utilizes attention modules in both the encoding and decoding paths to focus on relevant image regions and enhance localization accuracy. With the integration of attention mechanisms, the Attention UNet addresses the limitations of the traditional UNet in capturing long-range dependencies and handling class imbalance, resulting in more accurate segmentations."
    model_result = 'The AttUNet model in the Modern Lung Segmentation application achieved an Intersection over Union (IoU) of {:.3f} during testing, with a training time of {:.3f}s.'.format(test_iou, testing_time)

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