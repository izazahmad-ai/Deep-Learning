import streamlit as st

def margin_top(size):
    st.markdown("<div style='margin-top:{}px;'></div>".format(size), unsafe_allow_html=True)