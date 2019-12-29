# satellite_segmentation
This project for segmentation of satellite images from the dstl dataset from Kaggle(https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection). It uses a U net architecture for semantic segmentation. The dataset provides 16 band as well as 3 band images of high resolution. Only three band images were used the segmentation part due to the fact that for most satellite image datasets will not provide sixteen band images.

## Models
there are currently two supported models - 
1 - Unet from scratch
2 - Unet using vgg16 as encoder

## Dataset
Download the dataset from https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection in your current directory. and extract it. 
**NOTE** - the 16 band images can be deleted from the dataset because we are not using them.

### Preparing the dataset
After extracting the data run the following commands.
```bash
mv {extracted file name} dataset
python3 prep_data.py
```
### Configuring the model
The model can be configured using the config.json file

### Training
```bash
python3 train.py --config config.json
```

# Prediction
```bash
python3 predict.py --model {model path} --config config.json --img {image path} 
```
