# Object Detection in an Urban Environment

## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

[OPTIONAL] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. 

## Structure

### Data

The data you will use for training, validation and testing is organized as follow:
```
/home/workspace/data/waymo
    - training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test - contains 3 files to test your model and create inference videos
```

The `training_and_validation` folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling.

You will split this `training_and_validation` data into `train`, and `val` sets by completing and executing the `create_splits.py` file.

### Experiments
The experiments folder will be organized as follow:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment0/ - create a new folder for each experiment you run
    - experiment1/ - create a new folder for each experiment you run
    - experiment2/ - create a new folder for each experiment you run
    - label_map.pbtxt
    ...
```

## Prerequisites

### Local Setup

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.

### Download and process the data

**Note:** ”If you are using the classroom workspace, we have already completed the steps in the section for you. You can find the downloaded and processed files within the `/home/workspace/data/preprocessed_data/` directory. Check this out then proceed to the **Exploratory Data Analysis** part.

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file.

You can run the script using the following command:
```
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

You are downloading 100 files (unless you changed the `size` parameter) so be patient! Once the script is done, you can look inside your `data_dir` folder to see if the files have been downloaded and processed correctly.

### Classroom Workspace

In the classroom workspace, every library and package should already be installed in your environment. You will NOT need to make use of `gcloud` to download the images.

## Instructions

### Exploratory Data Analysis

You should use the data already present in `/home/workspace/data/waymo` directory to explore the dataset! This is the most important task of any machine learning project. To do so, open the `Exploratory Data Analysis` notebook. In this notebook, your first task will be to implement a `display_instances` function to display images and annotations using `matplotlib`. This should be very similar to the function you created during the course. Once you are done, feel free to spend more time exploring the data and report your findings. Report anything relevant about the dataset in the writeup.

Keep in mind that you should refer to this analysis to create the different spits (training, testing and validation).

### Create the training - validation splits
In the class, we talked about cross-validation and the importance of creating meaningful training and validation splits. For this project, you will have to create your own training and validation sets using the files located in `/home/workspace/data/waymo`. The `split` function in the `create_splits.py` file does the following:
* create three subfolders: `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`
* split the tf records files between these three folders by symbolically linking the files from `/home/workspace/data/waymo/` to `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`

Use the following command to run the script once your function is implemented:
```
python create_splits.py --data-dir /home/workspace/data
```

### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder. Now launch the training process:
* a training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`. You will report your findings in the writeup.

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it.

**Important:** If you are working on the workspace, your storage is limited. You may to delete the checkpoints files after each experiment. You should however keep the `tf.events` files located in the `train` and `eval` folder of your experiments. You can also keep the `saved_model` folder to create your videos.


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```

This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /data/waymo/testing/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```

## Submission 

### Project overview
This repository is for the first project of Udacity Self Driving Car Engineer Nanodegree. The goal of the project is Implementation of an Object Detection model for self-driving cars. In this project, we will utilize TF object detection API for better detection of objects on road, like cars, pedestrians, and cyclists. We will mainly use ResNet50 trained by Waymo Open dataset to detect and classify objects.

### Set up
For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the build directory of the starter code.

The instructions below are also contained within the build directory of the starter code. Requirements

NVIDIA GPU with the latest driver installed
docker / nvidia-docker
Build Build the image with:

docker build -t project-dev -f Dockerfile .
Create a container with:

docker run --gpus all -v <PATH TO LOCAL PROJECT FOLDER>:/app/project/ --network=host -ti project-dev bash
and any other flag you find useful to your system (eg, --shm-size).
Set up Once in container, you will need to install gsutil, which you can easily do by running:

curl https://sdk.cloud.google.com | bash
Once gsutil is installed and added to your path, you can auth using:

gcloud auth login
    
### Dataset
#### Dataset analysis

The waymo dataset mainly consists of 3 classes: vehicle, pedestrian and cyclist. Images in dataset containing examples from each class are shown below. The bounding-box for Vehicle is visualized in red, pedestrian in blue and cyclist in green.

![image](https://user-images.githubusercontent.com/94186015/202640883-b8d04ec0-92f8-4cc1-ac40-eeb6342e0845.png)

Dataset analysis is performed on num1=2000 random images in Exploratory Data Analysis file,the distribution of the 3 classes in the Dataset is visualized as follows.

![221118b](https://user-images.githubusercontent.com/94186015/202746493-5766fa1c-0abf-4eb4-8b8a-b2d587d7d6a9.PNG)
    
The distribution of the car object class versus the bicycle and pedestrian object classes in the dataset(num2=3000 images) is visualized as follows. It is titled "Distribution of Object Classes". 
    
![221118c](https://user-images.githubusercontent.com/94186015/202749102-7a2ff962-7b0f-47b5-b356-b3143605f3f0.PNG)
    

To estimate the day/night distribution of images in the dataset, the distribution of average brightness for each image in the dataset. The "Distribution of Image brightness " is illustrated as following. The majority of the images fall roughly within the middle range of 80-120 out of average brightness . Images in this range are daytime images. A small minority of the images were taken at night are in the 10-20 out of average brightness .
    
 ![221118d](https://user-images.githubusercontent.com/94186015/202749213-2c43c0b4-0cad-4823-b640-fe4561b36452.PNG)
   

#### Cross validation
The creation of training and validation split was already done in the workspace, with 86 images and 10 images in the training and validation set.

### Training
#### Reference experiment

The reference model is ResNet50 without augumentaion (see details of model parameters in experiments/reference/pipeline_new.config). Training loss of the model is shown as follows:

![220908](https://user-images.githubusercontent.com/94186015/202697580-6c1a752e-ca58-4e5c-b34c-7e1f9643d6de.PNG)

![220908c](https://user-images.githubusercontent.com/94186015/202700130-51b4d785-d918-4d38-8ea4-45ea92414a4b.PNG)


Precision:

![220908a](https://user-images.githubusercontent.com/94186015/202697927-be17fdb9-8331-4015-96dd-ece52807e0df.PNG)

Recall:

![220908b](https://user-images.githubusercontent.com/94186015/202698065-df957bfe-a65d-457b-9a12-f1d60c78a3cb.PNG)

From the plots, it is obvious that both losses are very noisy, especially localization loss. And localization loss does not seem to converge. Precision and Recall are extremly low and the model can barely detect and classify any object.

#### Improve on the reference

1.Increase the batch size from 2 to 6: batch size of 2 is too low for regular training of a large-size CNN like ResNet50. The detailed pipeline is in experiments/test1/pipeline_new.config. The results are as follows.
Training and validation loss of the model:

![220909](https://user-images.githubusercontent.com/94186015/202698612-1cac7422-9551-454d-9880-7d02574d67ab.PNG)

![220909c](https://user-images.githubusercontent.com/94186015/202700552-944b36e5-49cf-4af1-ad7c-a8d6b441fd24.PNG)


Precision:

![220909a1](https://user-images.githubusercontent.com/94186015/202698821-6e97822e-a3a2-4392-ade5-179f1fd9bc33.PNG)

Recall:

![220909b](https://user-images.githubusercontent.com/94186015/202700704-c9b29621-ff07-4445-933f-16515faedbcd.PNG)

We see some improvement in model loss, and Precision-Recall rate. This is a indication of better performance.

![2022-11-17_214405](https://user-images.githubusercontent.com/94186015/202701337-fd8fe446-307f-4107-a5c2-ed0afef81e5c.png)

2.Augmentation
Agumentation can be found in Explore augmentations.ipynb, and the detailed pipeline is in experiments/experiment2/pipeline_new.config. However, due to limitation of memory in the VM workspace, we have to resort to batch size of 2, and step size of 2500, which is very likely not enough for the network to converge. As a result, the performance does improve a lot, compared with reference model.

The results are as follows.

Training and validation loss of the model 

![220909AA](https://user-images.githubusercontent.com/94186015/202702653-34e48686-5260-465f-a187-bada79699de2.PNG)

![220909A4](https://user-images.githubusercontent.com/94186015/202703206-1696e30a-a752-4452-ba59-bada1f59780b.PNG)


Precision

![220909A2](https://user-images.githubusercontent.com/94186015/202702935-9128db8a-0692-4471-9a33-bd89f78f6661.PNG)

Recall

![220909A3](https://user-images.githubusercontent.com/94186015/202703005-feebf53d-6e0b-4e36-a195-14ca63fd15aa.PNG)



Althought we see a decrease in model loss, increase in precision and recall is tiny. The inference result is almost the same as that of the reference model, which barely detect anything. Thus, there is no pointing showing the inference video here.

![221118](https://user-images.githubusercontent.com/94186015/202716896-b486625f-553c-4fdc-b011-9eaac636b3d7.PNG)


By investigating the model on the test dataset, we can see the model is not goot at detecting small objects in the images. As cyclists only appear very scarcely in the datasets, we can expect the model to struggle to detect cyclists. In the future, improvements can be made in using higher resolution data for training, and sampling with more images with cyclist. More importantly, we want train for more steps with lower learning rate so that the model converges, provided that computers have larger computational resources and memories.











