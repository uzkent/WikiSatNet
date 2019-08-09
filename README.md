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
    python im2text_matching.py --lr 1e-4 --cv_dir {path} --batch_size 128 --train_csv {path} --val_csv {path}
```
If we initialize the weights by pre-training on ImageNet, the training step takes only 2 epochs. However, initializing weights **randomly** increases the number of epochs to 15. Our current code uses the model pre-trained on **ImageNet**.

At the end of training for 2 epochs, we can see the cosine loss going down from ~1 to ~0.35. We save the checkpoints and perform transfer learning experiments.

## Download DenseNet121 Model Pre-trained on WikiSatNet
You can download our __KERAS__ model pre-trained on **WikiSatNet** [here](https://drive.google.com/open?id=1Q69nGbhXFYoeJlgge-UPTZRVP0oL7Uy8). You can use this model for the downstream tasks that involves analyzing satellite images. It should provide significant boost especially with the small size dataset in the target task. For more details, you can check our paper. Soon, we will share the model trained in PyTorch with the code in this repository.

## Transfer Learning on the functional Map of the World (fMoW) Dataset
As a downstream task, we use the [fMoW](https://github.com/fMoW/dataset) dataset. It includes about **350k** training images, together with **50k** validation and test images. We pre-process the articles using the guidelines in the repository of the dataset. Similarly to the pre-training step, we form a csv file in the following format:
```
    Class Label, Image Location
```
Once we created the csv file for training and validation steps, we save it into the *data* directory.

Using **DenseNet161** model pre-trained on **ImageNet** we achieve **68.7** classification accuracy on the *temporal views*. On the other hand, we achieve **73.1** classification accuracy by using the model pre-trained on **WikiSatNet** with image to text matching. More importantly, when the number of training samples on the target task is reduced to **10k** labels, the model pre-trained on WikiSatNet outperforms ImageNet pre-training by **10%**.

To perform transfer learning on the fMoW dataset, you can use the following commands:
```
  python transfer_learning.py --lr 1e-4 --cv_dir {path} --batch_size 128 --load {path_to_checkpoints} --train_csv {path} --val_csv {path}
```

To cite our paper:
```
@article{uzkent2019learning,
  title={Learning to Interpret Satellite Images in Global Scale Using Wikipedia},
  author={Uzkent, Burak and Sheehan, Evan and Meng, Chenlin and Tang, Zhongyi and Burke, Marshall and Lobell, David and Ermon, Stefano},
  journal={arXiv preprint arXiv:1905.02506},
  year={2019}
}
```
More details to be posted soon.
