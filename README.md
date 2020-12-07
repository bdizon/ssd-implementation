# Pytorch Implementation of Single Shot MultiBox Detector (SSD)

This is an implementation of SSD300 with VGG16 backbone Architecture: [paper](https://arxiv.org/abs/1512.02325).

## Installation

* Install [Pytorch](https://pytorch.org/)
* Clone this repository (only support Python 3+)
* Download VOC dataset (currently support [VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/))
* Install requirements:
```
pip install -r requirements.txt
```
```
## Data
```
Data is from  the publicly available NWPU-crowd dataset. 
```

## Training
```
python train.py 
```
Parameters like decay, learning rate, and momentum are hard coded into the train.py file. 

## Evaluation
To evaluate a trained model:
```
python eval.py 
```
Default min_score of 0.01, an NMS max_overlap of 0.01, and top_k of 100 was used.

## Performance


Some examples in folder `images`
