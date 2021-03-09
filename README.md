# Dog classifier
The goal of this project is to create a dog classifier using state-of-the-art models for classification and localization. The pipeline processes real-world, user-supplied images. First, it tries to detect human faces. If supplied an image of a human face, the code will identify the resembling dog breed. Else, the algorithm will identify an estimate of the canine’s breed using a CNN.

## Detect Humans
I used OpenCV's implementation of [Haar feature-based cascade classifiers](https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html) to detect human faces in images. OpenCV provides many pre-trained face detectors, stored as XML files. I downloaded one of those and stored it in the ```haarcascades``` directory.

## Detect Dogs
I downloaded [VGG-16 pre-trained model](https://pytorch.org/vision/0.8/models.html) to detect dogs in images. I also explored other pre-trained networks such as Inception-v3 and ResNet-50. In order to check if an image contains a dog with the VGG-16 model, we need to check if the pre-trained model predicts an index between 151 and 268 (categories from 'Chihuahua' to 'Mexican hairless').

## Create a CNN to Classify Dog Breeds (from scratch)
I created a CNN that classifies dog breeds from scratch. The neural network has three convolutional layers with max pooling and ReLU activation. I used data augmentation with random horizontal flip and random rotation.

The model achieves 15% accuracy for the classification task of 133 different dog breeds.

## Create a CNN to Classify Dog Breeds (using transfer learning)
I used transfer learning to improve the model's performance. The dataset is small and similar to ImageNet (there are a lot of image of dogs in ImageNet) therefore I chose to get the convolutional layers from VGG16 fixed and train my own classifier on top.

The model achieves 72% accuracy for the classification task of 133 different dog breeds.

## Data
- Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
- Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz)
