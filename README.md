# Font Classification

This repo contains the code and models related to my paper [Convolutional Neural Networks for Font Classification](https://arxiv.org/abs/1708.03669).  I use ResNet-50 models to classify 227x227 crops of document images into font classes.  There are two sets of classes: Arabic fonts (40 in [KAFD dataset](http://kafd.ideas2serve.net/)) and 12 handwritten [scribal scripts](http://clamm.irht.cnrs.fr/script-classes/). 

For Arabic fonts, the model gets 99.2%, 98.8%, 97.9% accuracy on pages, lines, and 227x227 patches respectively.  For the 12 scribal script classes, we get 84.5% accuracy.  There is also a model that will classify medieval manuscripts into date ranges.

I have submitted similar code and models to the 2016 and 2017 [Classification of Latin Medieval Manuscripts (CLaMM) competitions](http://clamm.irht.cnrs.fr/).  [2016 code](https://github.com/ctensmeyer/clamm_submission) and [2017 code](https://github.com/ctensmeyer/clamm_2017).

This code depends on a number of python libraries: numpy, scipy, cv2 (python wrapper for opencv), and caffe [(my custom fork)](https://github.com/ctensmeyer/caffe).

## Docker

For those who don't want to install the dependencies, I have created a docker image to run this code. You must have the [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) plugin installed to use it though you can still run our models on CPU (not recommended).

The usage for the docker container is

```
nvidia-docker run -v $HOST_WORK_DIRECTORY:/data tensmeyerc/icdar2017:font python classify_latin_scripts.py /data/image_directory /data/predictions.csv [scripts|dates] $DEVICE_ID
nvidia-docker run -v $HOST_WORK_DIRECTORY:/data tensmeyerc/icdar2017:font python classify_arabic_fonts.py /data/image_directory /data/predictions.csv [lines|pages] $DEVICE_ID
```

`$HOST_WORK_DIRECTORY` is a directory on your machine that is mounted on /data inside of the docker container (using -v).  It's the only way to expose images to the docker container.
`$DEVICE_ID` is the ID of the GPU you want to use (typically 0).  If omitted, then the models are run in CPU mode.
There is no need to download the containers ahead of time.  If you have docker and nvidia-docker installed, running the above commands will pull the docker image (~2GB) if it has not been previously pulled.

Each python script has an option for what model to load, e.g., the model that predicts 'scripts' or the one that predicts 'dates' for medieval manuscripts.  For the Arabic fonts, the 'lines' model was trained on the line images of KAFD and the 'pages' model was trained on full pages.

## Citation

If you find this code useful for your research, please cite my paper:

```
@inproceedings{tensmeyer2017_binarization,
  title={Convolutional Neural Networks for Font Classification},
  author={Tensmeyer, Chris and Saunders, Daniel and Martinez, Tony},
  booktitle={ICDAR},
  year={2017},
  organization={IEEE}
}
```

