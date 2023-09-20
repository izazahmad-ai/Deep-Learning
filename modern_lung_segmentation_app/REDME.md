# Modern Lung Segmentation - Documentation

![Example Tutorial Modern Lung Segmentation](tutor_modern_lung_segmentation.gif)

## Brief Description

Modern Lung Segmentation is an innovative application designed to perform automatic lung segmentation on Chest X-Ray images. This application combines the power of deep learning with a simple and intuitive user interface, allowing users to easily perform fast and accurate lung segmentation.

## Steps to Create the Application

1. Preprocessing Chest X-Ray Dataset
   - The Chest X-Ray dataset from Kaggle is processed using advanced preprocessing techniques to prepare the data before training the models.

2. Building Deep Learning Models for Segmentation
   - Powerful and innovative segmentation models are developed in this application. There are four available models: UNet, UNet++, Attention UNet, and R2UNet. These models have proven to provide accurate segmentation results.

3. Model Training
   - The segmentation models are trained using the preprocessed Chest X-Ray dataset. Training is conducted to optimize the model's parameters and produce accurate and reliable segmentations.

4. Training Evaluation
   - Evaluation is performed using the Intersection over Union (IoU) metric, which is a commonly used measure to evaluate segmentation quality. Additionally, observations on training time and the number of model parameters are made to provide a comprehensive understanding of the model's performance.

5. Model Saving
   - After training is complete, the trained segmentation models are saved for future use in the Modern Lung Segmentation application.

6. Creating a User Interface (UI) with Streamlit
   - The application utilizes Streamlit, a framework for creating interactive UIs, to develop a simple and user-friendly interface.
   - The features offered by the application are designed to provide a smooth and efficient user experience.

## Key Features of the Application

1. Automatic Segmentation with Four Models
   - The application provides four pre-trained segmentation models: UNet, UNet++, Attention UNet, and R2UNet.
   - Users can choose the model that best suits their needs to perform automatic lung segmentation on Chest X-Ray images.

2. Input via Camera and File Upload
   - The application supports two types of input: users can either capture images directly through their device's camera or upload existing image files.

3. Download Lung Segmentation Results
   - After performing segmentation on Chest X-Ray images, users can download the generated lung segmentation results provided by the application.
   - These segmentation results contain valuable information and can be used for further analysis.

4. Model Explanation and Evaluation Menu
   - The application provides an explanation menu for each available segmentation model. Users can learn the technical details of each model and understand how they work.
   - Additionally, each model has an evaluation display that provides information about the quality of the segmentation produced by that model.

5. Model Comparison with Informative Charts
   - The application includes a model comparison menu that showcases segmentation models trained using the Chest X-Ray dataset.
   - This menu presents informative charts that include IoU evaluation results, computational time, and the number of model parameters.
   - The charts help users choose the most suitable model for their segmentation predictions.

6. Analysis of Evaluation Results
   - The evaluation results obtained from the segmentation models can be further analyzed to gain deeper insights into model performance and the quality of the generated segmentations.
   - This analysis provides a better understanding of the strengths and weaknesses of each model.

## Contributions and Future Development
We invite you to contribute to the development of the Modern Lung Segmentation application. You can add new features, improve existing segmentation models, or expand the dataset used.
For more information on contributing and development, please refer to our GitHub repository.

## Contact Us
If you have any questions, suggestions, or feedback, please feel free to reach out to our team at izazkhattak7@gmail.com.

We appreciate your interest in the Modern Lung Segmentation application and look forward to your involvement in improving healthcare technology.

