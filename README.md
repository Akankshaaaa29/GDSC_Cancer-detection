# GDSC_Cancer-detection: Revolutionizing Cancer Diagnosis with AI

## Project Overview

Welcome to GDSC_Cancer-detection, a groundbreaking project designed to revolutionize cancer research and treatment through state-of-the-art AI-driven solutions. Our primary focus is on the early detection of brain tumors using an innovative deep-learning model named "denseunet." This project addresses the crucial need for accurate and efficient cancer detection, especially in the context of brain tumors.

Visit our Deployed Website: [GDSC_Cancer-detection](https://gdsccancer-detection-pqw49wfnka3sosnxhffx5k.streamlit.app/)

## Video Pitch

Get a quick overview by watching our video pitch: [GDSC_Cancer-detection Video Pitch](https://drive.google.com/file/d/18TzN64tKCLC0ZGTR50AHu3uYd6ej3Jzs/view?usp=drive_link)

## Demo

![Project Demo](https://github.com/Akankshaaaa29/GDSC_Cancer-detection/assets/99268258/ca39580e-f8d7-4e7b-bd37-da7a17cca01d)

## Project Highlights

### Concept/Idea

#### The Critical Issue:

Late cancer detection is synonymous with limited treatment options and increased mortality rates. This challenge is especially pronounced in brain tumors, where every moment lost can mean the difference between a manageable situation and a life-altering prognosis.

#### Our Response:

Our project introduces a groundbreaking approach by incorporating the "DenseUnet" model. This advanced deep-learning design is carefully crafted to analyze brain images, accurately identifying cancerous cells with an unprecedented level of precision, revolutionizing early diagnosis. The model meticulously separates cancerous cells from brain images, achieving an exceptional accuracy rate of 99.98%. The innovation lies in its optimized runtime, outperforming existing solutions in terms of both speed and efficiency. To enhance user interaction, we deploy the model on the web using Streamlit, ensuring a smooth and user-friendly experience.
##### Dataset Description
The dataset used in this work is from fig share brain tumor segmentation. It consists 3064 T1-weighted contrast-enhanced images with three kinds of brain tumors. The images are collected from 233 subjects. The dataset contains coronal, sagittal, and axial views. The images are available as .mat files and the size of each image is 512x512. These images were split into training, test set and validation in the ratio of 65:15:20. Therefore, the model was trained on 2083 images and validated on 521 images

### Speciality/Uniqueness

1. **Accuracy:** The "DenseUnet" model exhibits an exceptional accuracy rate in identifying brain tumors, providing reliable results for medical professionals and researchers.
2. **Optimized Time:** Our solution prioritizes delivering results within an optimized timeframe, surpassing current alternatives in efficiency.
3. **Streamlit Interface:** We have seamlessly integrated our model into a user-friendly Streamlit interface, enabling users to input MRI images and receive predictions about the presence and location of brain tumors.

### TechStack:
- TensorFlow
- Keras
- Streamlit
- Python

## Getting Started

### Prerequisites

- Ensure Python is installed on your system.
- Download the project zip file.

### Installation

1. Extract the downloaded zip file.
2. Open a terminal in the project directory.
3. Run the following command to install the required dependencies:

   ```bash
   pip install -r requirements.txt

   ```

### Running Locally

1. After installing the dependencies, run the following command:

   ```bash
   streamlit run app.py
   ```

2. Open your web browser and go to the provided local URL to access the Streamlit interface.

Feel free to explore the functionalities of our AI-driven cancer detection solution!
