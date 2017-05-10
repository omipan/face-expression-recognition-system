# face-expression-recognition-system

The system is coded in tensorflow and python, its trained with a dataset of approximately 40000 images from Kaggle's Facial Expression Recognition Competition ( https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data ). The goal of the Competition was to classify the emotion displayed by the input image as 1 out of 7 possible emotions: anger, disgust, fear, happiness, sadness, surprise, neutral.

The input images with resolution 48x48 are first fed through 3 Convolutional Layers. After each of these layers, a max-pooling operation is performed on the output of the convolutional layer.
The output of the last convolutional layer (after the max-pooling operation) is first flattened to a single dimension (i.e. vector form) and is fed through two linear (hidden) layers. After each of these layers a rectified linear (ReLU) operation is performed. Finally, the output of the hidden layers is fed through another linear layer, which outputs 7 different values, corresponding to the 7 different emotions that the Kaggle Competition Dataset (which is used to train this model) defined. After proper optimization of the learning rate of our loss optimizer (Adam) we've come up with our final models which had ~63% accuracy, which is quite encouraging, taking into consideration the multiple classes and the similarity between some face expressions (face angles).


