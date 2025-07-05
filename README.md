# Gender & Age Prediction from Images

A deep learning-powered web app that predicts a person’s **gender** and **age** from a facial image. Built with **Keras** and **Streamlit** for real-time predictions and an intuitive UI.

## Project Objective 

The goal of this project is to apply deep learning techniques to real-world problems involving image data. Specifically, this project predicts a person’s gender and age from facial images using a convolutional neural network (CNN). It demonstrates multi-task learning with two outputs: classification (gender) and regression (age).

##  Model Architecture

The model is a Convolutional Neural Network (CNN) that processes 128x128 grayscale facial images and outputs:
- **Gender**: binary classification using either sigmoid or softmax
- **Age**: regression output
Basic architecture:
- Conv2D → ReLU → MaxPooling
- Conv2D → ReLU → MaxPooling
- Flatten
- Dense → ReLU
- Output1: Dense(1) with sigmoid/softmax (for gender)
- Output2: Dense(1) with linear activation (for age)

##  Features
- 📷 Upload a facial image (JPG, PNG)
-  Predicts:
  - **Gender** (Male/Female)
  - **Age** (Estimated in years)
- Supports both `sigmoid` and `softmax` gender output models
- Displays model confidence (for sigmoid)
- Lightweight and fast with real-time feedback

##  Project files
gender-age-predictor/
- app.py
- model.h5
- requirements.txt
- README.md   

## how to run this 
Step 1: Clone the project
-git clone https://github.com/your-username/gender-age-predictor.git

-cd gender-age-predictor

Step 2: Install required packages	
pip install -r requirements.txt

Step 3: Add your model
Put your model.h5 file in the main folder.
The model should:
- Take input of size 128x128 grayscale image
- Give 2 outputs:
1. Gender (0 or 1, or [probabilities])
2. Age (a number)

## Run the Web App
  streamlit run app.py
  
  Then go to your browser and open: http://localhost:8501

## Model Details
- Input Image: 128x128 pixels, grayscale
- Gender Output:
  - sigmoid: value between 0–1 (0 = Male, 1 = Female)
  - softmax: [Male %, Female %]
- Age Output: Number (like 22.5

## Datasets You Can Use for Training
- UTKFace Dataset: https://susanqq.github.io/UTKFace/

## Requirements
- Python 3.7 or above
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow
Install everything using:
    pip install -r requirements.txt

##  Experiments & Observations

- Models trained with sigmoid performed better with confidence display.
- Age prediction was more accurate on younger faces.
- Overfitting observed after 30 epochs without regularization.

## License
This project is free to use under the MIT License

## Live App:
[https://your-streamlit-url.streamlit.app](https://gender-age-prediction-attfkzgxlgqh9z4dewgsyp.streamlit.app/)

## Contribute
Want to help improve the app?
Fork the repo, make changes, and send a pull request!

## Contact
If you have questions or ideas, open an issue on GitHub or start a discussion.
 email: daliadayna101@gmail.com
