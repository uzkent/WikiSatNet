# Learning to Interpret Satellite Images Using Wikipedia
This repository contains the implementation of our IJCAI 2019 paper __Learning to Interpret Satellite Images Using Wikipedia__. [arxiv](https://arxiv.org/pdf/1905.02506.pdf)

Unfortunately, as of now, we are still negotiating with DigitalGlobe to purchase the images for WikiSatNet. However, we plan on releasing the model pre-trained on WikiSatNet that boosts the accuracy on the dowstream tasks. Additionally, we are working on learning visual features with no labels using publicly-available Sentinel-2 images. More updates on this will be posted soon.

## Process the Geolocated Articles using Doc2Vec
You can find how to process geolocated articles using Doc2Vec in this [repository](https://github.com/ermongroup/WikipediaPovertyMapping).

For each wikipedia article, we learn a __300-D__ textual embedding and save them in a file. We believe that this __300-D__ embedding is a *rich summary* of the corresponding __satellite image__. We then train the CNN to learn this summaries to learn robust, domain-specific features that can be highly useful
for transfer learning.

## Creating the CSV file
After processing articles using Doc2Vec, we form a csv file for both **training** and **validation** steps. The csv files are formatted as below.
```
    Embedding Location, Image Location, Size of the Image
    directory, directory, integer
```
Our dataset consists of 800k 1000x1000 pixels images. Additionally, we perform data augmentation by cropping the central 250x250, 500x500, 750x750, and 1000x1000 pixels areas resulting in 3200k images. Finally, we save the csv files into the **data** directory.

## Training
To train the CNN, we should use the following command:
```
    python im2text_matching.py --lr 1e-4 --cv_dir {path} --batch_size 128
```
If we initialize the weights by pre-training on ImageNet, the training step takes only 2 epochs. However, initializing weights **randomly** increases the number of epochs to 15. Our current code uses the model pre-trained on **ImageNet**.

At the end of training for 2 epochs, we can see the cosine loss going down from ~1 to ~0.35. We save the checkpoints and perform transfer learning experiments.
