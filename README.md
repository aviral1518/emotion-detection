# Emotion-detection

## Introduction

This project aims to classify the emotion on a person's face into one of **seven categories**, using deep convolutional neural networks. This repository is an implementation of [this](https://drive.google.com/file/d/1Jm9mI6V85XGD4_eKTje0voz1PKKVFOsr/view?usp=sharing) research paper. The model is trained on the **FER-2013** dataset which was published on International Conference on Machine Learning (ICML). This dataset consists of 35887 grayscale, 48x48 sized face images with **seven emotions** - angry, disgusted, fearful, happy, neutral, sad and surprised.

## Dependencies

* Python 3.6, [OpenCV 3 or 4](https://opencv.org/), [Tensorflow](https://www.tensorflow.org/), [TFlearn](http://tflearn.org/)
* To install the required packages, run `pip install -r requirements.txt`.

### TFLearn

* Download the **trained model** files from [here](https://drive.google.com/drive/folders/14XDIGAdNdMcpvWSn5yCvbdwUfHsYxWTh?usp=sharing), extract it and copy the files into the current working directory.

* To run the program to detect emotions only in **one face**, type `python model.py singleface`.

* To run the program to detect emotions on all faces close to camera, type `python model.py multiface`. Note that this sometimes generates incorrect predictions.

* The folder structure is of the form:  
  TFLearn:
  * emojis (folder)
  * `model.py` (file)
  * `multiface.py` (file)
  * `singleface.py` (file)
  * `model_1_atul.tflearn.data-00000-of-00001` (file)
  * `model_1_atul.tflearn.index` (file)
  * `model_1_atul.tflearn.meta` (file)
  * `haarcascade_frontalface_default.xml` (file)

## Algorithm

* First, we use **haar cascade** to detect faces in each frame of the webcam feed.

* The region of image containing the face is resized to **48x48** and is passed as input to the ConvNet.

* The network outputs a list of **softmax scores** for the seven classes.

* The emotion with maximum score is displayed on the screen.

## References

* "Challenges in Representation Learning: A report on three machine learning contests." I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B
   Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li,  
   X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu,
   M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and
   Y. Bengio. arXiv 2013.
