# Facial-Expression-Recognition

## The following images are a small sample from the training data
<img src="https://github.com/BibhuLamichhane/facial-expression-recognition/blob/master/sample.png">

## These are the predictions made by my model on some random images from the internet
<img src="https://github.com/BibhuLamichhane/facial-expression-recognition/blob/master/angry.jpg"><img src="https://github.com/BibhuLamichhane/facial-expression-recognition/blob/master/surprise.jpg"><img src="https://github.com/BibhuLamichhane/facial-expression-recognition/blob/master/sad.jpg">

## How to run the code 

### To predict the emotion in real time using your webcam
1. Create a python 3.6.5 virtual environment
2. Run the commands: 
```
      pip install -r requirements.txt
      python main.py
```

### If there were any error while using my weights you have to train your own model
1. Find a facial expression dataset (I would recommend <a href="https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data">Kaggle Facial Expression Recognition Dataset</a>)
2. In the file recognizer_model_trainer.py write the code to convert these images to (x, 48, 48, 1) shaped arrays and use the variable name "x1" to store the image pixels and "y1" to store the the labels of the image
3. Run the file recognizer_model_trainer.py
4. Run the file main.py

### To make the model predict the emotion in a image provided by the user run the command
      python predict.py
