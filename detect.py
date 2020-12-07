import sys
sys.path.append("./model/")
import os
from PIL import Image, ImageDraw, ImageFont
from utils import *
from torchvision import transforms
import torch
from model.SSD300 import SSD300
from model.vgg import VGG16BaseNet, AuxiliaryNet, PredictionNet
import argparse
import csv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_path = "images_part1/000079.png"
img_path = input_path

trained_model = torch.load(saved_model)
output_path = "results/000079.png"
min_score = 0.05
max_overlap = 0.3
top_k = 200

start_epoch = trained_model["epoch"] + 1
print('\nLoaded model trained with epoch %d.\n' % start_epoch)
model = trained_model['model']
model = model.to(device)
model.eval()
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
def detect(original_image, min_score, max_overlap, top_k, suppress = None):
    image = normalize(to_tensor(resize(original_image)))
    image = image.to(device)
    locs_pred, cls_pred = model(image.unsqueeze(0))
    detect_boxes, detect_labels, detect_scores = model.detect(locs_pred, cls_pred,
                                                              min_score, max_overlap, top_k)
    detect_boxes = detect_boxes[0].to('cpu')
    original_dims = torch.FloatTensor(
            [original_image.width, original_image.height, original_image.width,
             original_image.height]).unsqueeze(0)
    detect_boxes = detect_boxes * original_dims
    detect_labels = [rev_label_map[l] for l in detect_labels[0].to('cpu').tolist()]
    print(detect_labels)

    if detect_labels == ["background"]:
        print("original image returned")
        return original_image
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("arial.ttf", 15)
    for i in range(detect_boxes.size(0)):
        if suppress is not None:
            if detect_labels[i] in suppress:
                continue
        box_location = detect_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[detect_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=
                       label_color_map[detect_labels[i]])
        text_size = font.getsize(detect_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[detect_labels[i]])
        draw.text(xy=text_location, text=detect_labels[i].upper(), fill='white',font=font)
    print("Detect label: ", detect_labels)
    return annotated_image,len(detect_labels)
if __name__ == '__main__':
   
    directory = "images_part1"
    output_directory = "results"
    num = 0
    with open("new_test.csv") as f:
        num = 0
        err = 0;
        for i, line in enumerate(f):
            print(line)
            img_path = os.path.join('./',line.split(',')[0])
            num+=1
            output_path = os.path.join(output_directory, 'num'+str(num)+'.png')

            original_image = Image.open(img_path, mode='r')
            original_image = original_image.convert('RGB')
            annotated_image, predict_number = detect(original_image, min_score=min_score,max_overlap=max_overlap, top_k= top_k)
            annotated_image.save(output_path)
            annotations_path = os.path.join('./', line.split(',')[1].strip('\n'))
            annotation_dict = json.load(open(annotations_path, "r"))
            num_people = annotation_dict['num-people']
            err+=abs(num_people-predict_number)
            print(err,abs(num_people-predict_number) )

        print(err/num)

    img_path = "images_part1/028302.png"
    output_path = "results/028302.png"
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    annotated_image = detect(original_image, min_score=min_score, 
                            max_overlap=max_overlap, top_k= top_k)
    annotated_image.save(output_path)
