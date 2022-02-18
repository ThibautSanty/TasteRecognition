# TasteRecognition
A project that recognizes if you would like a product when you taste it or not. Here I compare a LSTM and a CNN model. The tutorial opens up a widow with your camera, where it looks for your face. Once the program found a face it will predict directly a face reaction with the CNN model. Only after 21 face reactions it will predict with the LSTM model.

### why
This project was original created to improve supermarkets with their stock. You know when you go to the store and you can taste some of their products. Well if the store looks at your face his reaction it could predict how much people that likes the product and how much that diskikes the product. That concludes if they would need to buy more stock of it or just stop selling that product.


### CNN (Convolutional neural network)
You may already know this but short explanation is an AI that predicts a result with 1 picture. So every picture it takes, it will predict 'like' or 'dislike'


### LSTM (Long short term memory)
This model will look at a sequence of data. Rather than predict on a single picture, it will create a sort of memory that can capture a reaction.


### conclusion
For this problem we want to capture a person his reaction, therefor using a LSTM seems better in this case. So we can see if a neutral face goes to dislike or goes to like.


## instructions
To test out the project you'll need python3.x.x -> download it on (https://www.python.org/downloads/)

Also you will need these libraries:
* opencv-python
* numpy
* keras

In case you don't have them already.
```
$ python pip3 install opencv-python numpy keras
```
