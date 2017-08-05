# Fine-grained Dog Classification competition
- This is a dog classification competition held by Baidu. Further information at http://js.baidu.com/

## Framework
- [Keras](https://keras.io/)
- [Tensorflow Backend](https://www.tensorflow.org/)

## Hardware
- Geforce GTX TITANX 12G
- Intel® Core™ i7-6700 CPU
- Memory 16G

## Data
- Download the images from Baidu Cloud
  - Training Set: http://pan.baidu.com/s/1slLOqBz Key: 5axb
  - Test set: http://pan.baidu.com/s/1gfaf9rt Key：fl5n
- Put the images into diffrent directory by their class labels. Refer to [altoFolders.py](preprocessing/altoFolders.py) for doing this.
- Take 20% of the labeled data for validation. Refer to [divforValidation.py](preprocessing/divforValidation.py).

## Base Model
- [VGG19](models/vgg19.py) for deep feature extraction,which is provided in keras models.
- Softmax for classification.

## Evaluate
- Predict the classes for unlabeled data one by one refering to [predict_onebyone](evaluate/predict_onebyone.py) and by generator refering to [predict_bygenerator.by](evaluate/predict_bygenerator.py).
## to be continued
> Feel free to contact me if you have any issues or better ideas about anything.

> by Holy