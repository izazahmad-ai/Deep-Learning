import streamlit as st
import functions as f
import components.components as comp
from pathlib import Path

# TITLE TAB
st.set_page_config(page_title="Compare & Evaluation Models 📊")

# HEADER
def header_page():
    st.markdown("<h1 style='text-align: center;'>Compare & Evaluation <br> Models 📊</h1>", unsafe_allow_html=True)

# FUNCTIONS
f.modelsCheck()
description_evaluation = Path('description_evaluation.md').read_text()

# BODY
def body_page():
    f.createPlotEvaluation(f.getEvaluationDF())
    st.markdown(description_evaluation)


# SIDEBAR
with st.sidebar:
    description = "Modern lung segmentation applications aim to identify and separate the lungs from radiological images, such as Chest X-Rays (CXRs). Lung segmentation improves the accuracy of image classification model training and assists medical teams in their observations. The accuracy level is measured using the Intersection over Union (IoU) method in several models that utilize a dataset containing 3000 images and their corresponding masks."

    st.markdown(f'<p style="font-size:15px; text-align:justify">{description}', unsafe_allow_html=True)
    st.markdown("<span style='font-size:17px; text-align:justify font-weight:bold;'>Model Specification : </span>", unsafe_allow_html=True)
    st.write(f.getEvaluationDF())


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
