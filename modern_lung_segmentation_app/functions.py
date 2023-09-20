import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
from skimage import filters
from PIL import Image
import numpy as np
import plotly.express as px
import os
import base64
import shutil
import gdown
import pandas as pd

def modelsCheck():
    # SET SESSION STATE
    st.session_state.existing_models = {
        'UNet' : '1zkvSyAAhG2zWZh2fLt-Gg5GFiDyo0X04',
        'UNet++' : '1--55QSoelNuMpzNmCTd_ksdCnTnZznWA',
        'R2UNet' : '1nuUI26UdTNWVVqZEiOV5CmjcSR7HP5UA',
        'AttUNet' : '1aIKJpKm_SOZWmo9Mqp2z97np16Fs2txq',
        }
    models_and_url_id = st.session_state.existing_models
    isModelsExist = []
    for model_name in models_and_url_id.keys():
        folder_path = 'models/{}/'.format(model_name)
        model_filename = model_name + '.h5'
        isModelExist = os.path.isfile(os.path.join(folder_path, model_filename))
        isModelsExist.append(isModelExist)
    
    if False in isModelsExist:
        # HAVE TO UPDATE THE URL IF MODEL UPDATED IN COLAB
        for name, model_url_id in zip(models_and_url_id.keys(), models_and_url_id.values()):
            save_path = 'models/{}/{}.h5'.format(name, name)
            url = "https://drive.google.com/uc?id={}".format(model_url_id)
            gdown.download(url, save_path, quiet=False)

def refreshImagesInFolder():
    folder_path = 'results_images'
    shutil.rmtree(folder_path, ignore_errors=True)
    os.makedirs(folder_path, exist_ok=True)
    model_folders = st.session_state.existing_models.keys()
    image_folders = ["ori_images", "mask_images", "line_mask_images", "merge_images", "segmented_images"]

    for model_folder in model_folders:
        model_folder_path = os.path.join(folder_path, model_folder)
        os.makedirs(model_folder_path, exist_ok=True)
        for image_folder in image_folders:
            image_folder_path = os.path.join(model_folder_path, image_folder)
            os.makedirs(image_folder_path, exist_ok=True)

def createSegmentation(model, image_arr):
    # Image Array
    image_arr_reshape = image_arr[np.newaxis, ...] # Reshape menjadi (1, width, height, channel) agar bisa predict

    # Model Predict Tensorflow
    mask_arr = model.predict(image_arr_reshape, verbose=0)

    # Create Segmented Images
    segmented_image_arr = image_arr * mask_arr[0]

    # Create Line Segmentation
    line_mask_arr = filters.sobel(mask_arr[0])

    # Convert Image to uint8
    image_arr_uint8 = (image_arr*255).astype(np.uint8)
    line_mask_arr_uint8 = (line_mask_arr*255).astype(np.uint8)
    
    # Convert Image to PIL
    image_pil = Image.fromarray(image_arr_uint8[:, :, 0])
    line_mask_pil = Image.fromarray(line_mask_arr_uint8[:, :, 0])

    # Convert image to RGBA
    line_mask_pil_rgba = line_mask_pil.convert("RGBA")
    image_pil_rgba = image_pil.convert("RGBA")

    # Convert image to tuple data
    line_mask_pil_data = line_mask_pil_rgba.getdata()
    image_pil_data = image_pil_rgba.getdata()

    image_result = []
    for mask_line_channels, image_channels in zip(line_mask_pil_data, image_pil_data):
        if mask_line_channels != (0, 0, 0, 255):
            image_channels = list(image_channels)
            image_channels[0] = int((image_channels[0] + 180) / 2)
            image_channels[1] = int((image_channels[1] + 0) / 2)
            image_channels[2] = int((image_channels[2] + 0) / 2)
            # image_channels[3] = 120
            image_channels = tuple(image_channels)
            image_channels = image_channels
            image_result.append(image_channels)
        else:
            image_result.append(image_channels)
    image_pil_rgba.putdata(image_result)

    IMAGE = array_to_img(image_arr)
    MASK = array_to_img(mask_arr[0])
    LINE_MASK = image_pil_rgba
    SEGMENTED_IMAGE = array_to_img(segmented_image_arr)
    
    MERGE_IMAGE = Image.new('RGB', (256*3, 256))
    MERGE_IMAGE.paste(IMAGE, (0, 0))
    MERGE_IMAGE.paste(LINE_MASK, (256, 0))
    MERGE_IMAGE.paste(MASK, (256*2, 0))

    return IMAGE, MASK, LINE_MASK, MERGE_IMAGE, SEGMENTED_IMAGE

def getArrImages():
    images_arr = []
    if st.session_state.input_selected == 'camera':
        image_session = st.session_state.images
        image =  load_img(image_session, color_mode='grayscale')
        width, height = image.size
        # Set coordinat to crop image
        left = (width/2) - (height/2)
        top = 0
        right = (width/2) + (height/2)
        bottom = height
        # Cut image at center coordinate
        cropped_image = image.crop((left, top, right, bottom))
        cropped_image_resize =  cropped_image.resize((256, 256))
        image_arr = img_to_array(cropped_image_resize).astype('float32') / 255.0
        images_arr.append(image_arr)
    else:
        images_session = st.session_state.images
        for image_session in images_session:
            image =  load_img(image_session, color_mode='grayscale', target_size=(256, 256))
            image_arr = img_to_array(image).astype('float32') / 255.0
            images_arr.append(image_arr)

    return np.array(images_arr)

def saveSegmentation():
    # Refresh or Delete All Images In Directory
    refreshImagesInFolder()
    # Get Images Array
    images_arr = getArrImages()
    # Get Used Model from Session
    used_model = st.session_state.used_model
    if used_model == 'All Models At Once':
        models = st.session_state.existing_models.keys()
        for model_name in models:
            model = tf.keras.models.load_model('./models/{}/{}.h5'.format(model_name, model_name), compile=False)
            for i, image_arr in enumerate(images_arr):
                ori_image, mask_image, line_mask_image, merge_image, segmented_image = createSegmentation(model, image_arr)
                # Save Image to Directory
                _ = ori_image.save("results_images/{}/ori_images/".format(model_name) + "{}_ori_{}".format(model_name, i+1)+".bmp") # image
                _ = mask_image.save("results_images/{}/mask_images/".format(model_name) + "{}_mask_{}".format(model_name, i+1)+".bmp")  # mask
                _ = line_mask_image.save("results_images/{}/line_mask_images/".format(model_name) + "{}_line_{}".format(model_name, i+1)+".bmp")  # line_mask
                _ = merge_image.save("results_images/{}/merge_images/".format(model_name) + "{}_merge_{}".format(model_name, i+1)+".bmp")  # merge_image
                _ = segmented_image.save("results_images/{}/segmented_images/".format(model_name) + "{}_segmented_{}".format(model_name, i+1)+".bmp")  # merge_image
            
    else:
        model_name = used_model.split()[0]
        model = tf.keras.models.load_model('./models/{}/{}.h5'.format(model_name, model_name), compile=False)
        for i, image_arr in enumerate(images_arr):
            ori_image, mask_image, line_mask_image, merge_image, segmented_image = createSegmentation(model, image_arr)
            # Save Image to Directory
            _ = ori_image.save("results_images/{}/ori_images/".format(model_name) + "{}_ori_{}".format(model_name, i+1)+".bmp") # image
            _ = mask_image.save("results_images/{}/mask_images/".format(model_name) + "{}_mask_{}".format(model_name, i+1)+".bmp")  # mask
            _ = line_mask_image.save("results_images/{}/line_mask_images/".format(model_name) + "{}_line_{}".format(model_name, i+1)+".bmp")  # line_mask
            _ = merge_image.save("results_images/{}/merge_images/".format(model_name) + "{}_merge_{}".format(model_name, i+1)+".bmp")  # merge_image
            _ = segmented_image.save("results_images/{}/segmented_images/".format(model_name) + "{}_segmented_{}".format(model_name, i+1)+".bmp")  # segmented_image

def showImageMarkdown(path):
    image = open(path, "rb") 
    contents = image.read() 
    data_url = base64.b64encode(contents).decode("utf-8") 
    image.close()
    st.markdown( f'<p style="text-align: center;"><img src="data:image/gif;base64,{data_url}" alt="results_images"></p>', unsafe_allow_html=True)

def showSegmentationFromCamera():
    # Get Used Model from Session
    used_model = st.session_state.used_model
    # Process All Models At Once
    if used_model == 'All Models At Once':
        model_names = st.session_state.existing_models.keys()
        for model_name in model_names:
            # MODEL TITLE
            st.markdown("<h4 style='text-align: center; margin-top:15px'>Segmented Images With {}</h4>".format(model_name), unsafe_allow_html=True)
            image_path = "results_images/{}/segmented_images/".format(model_name) + "{}_segmented_{}".format(model_name, 1)+".bmp"
            buff1, col, buff2 = st.columns([1, 1, 1])
            # SHOW RESULTS OF IMAGES
            with col:
                showImageMarkdown(image_path)
            buff1, col, buff2 = st.columns([2.3, 2, 2])
            # SHOW DOWNLOAD
            with col:
                downloadSegmentedImage(model_name)

    # Process 1 Model
    else:
        model_name = used_model.split()[0]
        # MODEL TITLE
        st.markdown("<h4 style='text-align: center; margin-top:15px'>Segmented Images With {}</h4>".format(model_name), unsafe_allow_html=True)
        image_path = "results_images/{}/segmented_images/".format(model_name) + "{}_segmented_{}".format(model_name, 1)+".bmp"
        buff1, col, buff2 = st.columns([1, 1, 1])
        # SHOW RESULTS OF IMAGES
        with col:
            showImageMarkdown(image_path)
        buff1, col, buff2 = st.columns([2.3, 2, 2])
        # SHOW DOWNLOAD
        with col:
            downloadSegmentedImage(model_name)

def showSegmentationFromFileUploader():
    # Get Used Model from Session
    used_model = st.session_state.used_model
    # Process All Models At Once
    if used_model == 'All Models At Once':
        model_names = st.session_state.existing_models.keys()
        for model_name in model_names:
            # MODEL TITLE
            st.markdown("<h4 style='text-align: center; margin-top:15px'>Segmented Images With {}</h4>".format(model_name), unsafe_allow_html=True)
            cols = st.columns([1, 1, 1])
            # SHOW RESULTS OF IMAGES
            for i in range(len(st.session_state.images)):
                if i == 3:
                    break
                with cols[i]:
                    image_path = "results_images/{}/segmented_images/".format(model_name) + "{}_segmented_{}".format(model_name, i+1)+".bmp"
                    showImageMarkdown(image_path)
            # SHOW DOWNLOAD
            buff1, col, buff2 = st.columns([2.4, 2, 2])
            with col:
                downloadSegmentationResults(model_name)
                
    else:
        model_name = used_model.split()[0]
        # MODEL TITLE
        st.markdown("<h4 style='text-align: center; margin-top:15px'>Segmented Images With {}</h4>".format(model_name), unsafe_allow_html=True)
        cols = st.columns([1, 1, 1])
        # SHOW RESULTS OF IMAGES
        for i in range(len(st.session_state.images)):
            if i == 3:
                break
            with cols[i]:
                image_path = "results_images/{}/segmented_images/".format(model_name) + "{}_segmented_{}".format(model_name, i+1)+".bmp"
                showImageMarkdown(image_path)
        # SHOW DOWNLOAD
        buff1, col, buff2 = st.columns([2.4, 2, 2])
        with col:
            downloadSegmentationResults(model_name)

def zippingImage(model_name):
    folder_path = 'results_images/{}'.format(model_name)
    save_zip_path = 'results_images/results_images_{}'.format(model_name)
    shutil.make_archive(save_zip_path, 'zip', folder_path)

def downloadSegmentedImage(model_name):
    st.download_button(label='Download Segmented Image {}'.format(model_name),
            data= open('results_images/{}/segmented_images/{}_segmented_1.bmp'.format(model_name, model_name), 'rb').read(),
            file_name='results_image_{}.bmp'.format(model_name),
            mime='image/bmp')

def downloadSegmentationResults(model_name):
    zippingImage(model_name)
    st.download_button(label='Download All Images {}'.format(model_name),
            data= open('results_images/results_images_{}.zip'.format(model_name), 'rb').read(),
            file_name='results_images_{}.zip'.format(model_name),
            mime='application/zip')

def getEvaluationDF():
    models = st.session_state.existing_models
    evaluation_df = pd.DataFrame(columns=['Model', 'IoU', 'Time (s)', 'Trainable Params (M)'])
    for model_name in models:
        # Get CSV
        model_csv_logger = pd.read_csv('./models/{}/csv_logger_{}.csv'.format(model_name, model_name))
        # Get IoU and Testing Timein CSV
        test_iou = float("{:.3f}".format(model_csv_logger.iloc[-1]['test_iou']))
        testing_time = float("{:.3f}".format(model_csv_logger.iloc[-1]['testing_time']))
        trainable_params = model_csv_logger.iloc[-1]['trainable_params']
        # Put model_name and ioU in DataFrame
        evaluation_df.loc[len(evaluation_df.index)] = [model_name, test_iou, testing_time, trainable_params]
    
    evaluation_df = evaluation_df.sort_values(by=['Trainable Params (M)'])
    # Start index from 1
    evaluation_df.index = np.arange(1, len(evaluation_df) + 1)

    return evaluation_df

def createPlotEvaluation(df):
    df_sorted = df.sort_values(by='IoU')
    # Plotting
    fig = px.scatter(df_sorted, x='Model', y='IoU', size='Time (s)', color='Trainable Params (M)',
                    hover_data=['Time (s)', 'Trainable Params (M)'], title='Model Evaluation')

    # Line plot
    fig.add_trace(px.line(df_sorted, x='Model', y='IoU').data[0])

    # Customize the plot
    fig.update_layout(
        xaxis=dict(title='Model'),
        yaxis=dict(title='IoU'),
        coloraxis=dict(colorbar=dict(title='Trainable Params (M)')),
    )

    # Show the plot
    st.plotly_chart(fig)
