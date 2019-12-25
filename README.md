# Natural Scene Classification

Dataset:
https://www.kaggle.com/puneet6060/intel-image-classification


After creating and testing a variety of models, I arrived on three models that function on this dataset.

I was not able to hit 90% accuracy on this dataset. These are the accuracies of the models I created:

ResNet Based Model - ~87% (Too large to put on GitHub and extremely slow)

MobileNetV2 Based Model - 82.3%

Custom Convolutional Neural Network - 86.43%

I personally tested the custom network on some of my own images, and I found it to be the most functional network. I was expecting this problem to be easily solved with transfer learning, but it turned out that a custom model is more accurate in the relatively small amount of training I did.
