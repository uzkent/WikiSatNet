# Learning to Interpret Satellite Images Using Wikipedia
This repository contains the implementation of our IJCAI 2019 paper __Learning to Interpret Satellite Images Using Wikipedia__.

Unfortunately, as of now, we are still negotiating with DigitalGlobe to purchase the images for WikiSatNet. However, we plan on releasing the model pre-trained on WikiSatNet that boosts the accuracy on the dowstream tasks. Additionally, we are working on learning visual features with no labels using publicly-available Sentinel-2 images. More updates on this will be posted soon.

## Process the Geolocated Articles using Doc2Vec
You can find how to process geolocated articles using Doc2Vec in this [repository](https://github.com/ermongroup/WikipediaPovertyMapping).

For each wikipedia article, we learn a __300-D__ textual embedding and save them in a file. We believe that this __300-D__ embedding is a *rich summary* of the corresponding __satellite image__. We then train the CNN to learn this summaries to learn robust, domain-specific features that can be highly useful
for transfer learning.
