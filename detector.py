import time
import torch 
import torch.nn as nn
from torch.autograd import Variable 
import numpy as np 
import cv2 
from utils import util 
import argparse 
import os 
from darknet import Darknet 
import pickle as pkl 
import pandas as pd 
import random 
from utils.util import *


def arg_parse():
    """
    Argument parser for detect module
    """
    parser = argparse.ArgumentParser(description="Yolov3 detection module")

    parser.add_argument("--images", dest= 'images', help=
                                "Image Directory containing images to perform detection upon",
                                default= "imgs", type=str)
    parser.add_argument("--det", dest="det", help=
                                "Image Directory to store detections to",
                                default="det", type=str)
    parser.add_argument("--bs", dest ="bs", help=
                                "Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help=
                                "Object confidence to filter predictions",
                                default= 0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help=
                                "NMS Threshold", default=0.4)
    parser.add_argument("--cfg", dest = "cfgfile", help="Config File",
                                    default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest = "weightsfile", help = "weights file",
                                default = "models/yolov3.weights", type = str)
    parser.add_argument("--reso", dest = "reso", help=
                                "Input resolution of the network. Increase to increase accuracy. Descrease to increase speed",
                                default="416", type=str)

    return parser.parse_args()

args = arg_parse() 
images = args.images 
batch_size = int(args.bs) 
confidence = float(args.confidence) 
nms_thres = float(args.nms_thresh)
start = 0 
CUDA = torch.cuda.is_available() 

num_classes = 80 # For coco 
classes = load_classes("data/coco.names")

# Setup the network 
print("Loading network...")
model = Darknet("cfg/yolov3.cfg")
model.load_weights(args.weightsfile)
print("Network weights loaded successfully.")

model.net_info["height"] = args.reso 
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

# If there is GPU, put model on GPU
if CUDA: 
    model.cuda() 

# Set the model in evaluation mode 
model.eval() 

# Read input images
read_dir = time.time() 
#Detection phase 
try: 
    imlist = [os.path.join(os.path.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = [] 
    imlist.append(os.path.join(os.path.realpath("."), images))
except FileNotFoundError:
    print(f"No file or directory with the name {images}")
    exit() 

if not os.path.exists(args.det):
    os.makedirs(args.det)

load_batch = time.time() 
loaded_ims = [cv2.imread(x) for x in imlist] 
# pytorch variable for images 
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

#List containing dimensions of original images
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

if CUDA:
    im_dim_list = im_dim_list.cuda()

leftover = 0
if (len(im_dim_list) % batch_size):
   leftover = 1

if batch_size != 1:
   num_batches = len(imlist) // batch_size + leftover            
   im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                       len(im_batches))]))  for i in range(num_batches)]  
write=0 
start_det_loop = time.time() 
if CUDA:
	im_dim_list = im_dim_list.cuda()
for i, batch in enumerate(im_batches):
    #load the image 
    start = time.time() 
    if CUDA:
        batch = batch.cuda() 
    
    prediction = model(Variable(batch), CUDA) 
    prediction = write_results(prediction, confidence, nms_conf=nms_thres, num_classes=80)
    end = time.time() 

    if type(prediction) == int: 
        for im_num, image in enumerate(imlist[i*batch_size: min((i+1) * batch_size, len(imlist))]):
            im_id = i * batch_size + im_num 
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("-------------------------------------------------------------")
        continue 
    prediction[:,0] += i* batch_size  #Transform the attribute from index in batch to index in imlist 

    if not write:
        output = prediction 
        write = 1 
    else: 
        output = torch.cat((output, prediction))

    for im_num, image in enumerate(imlist[i*batch_size: min((i+1) * batch_size, len(imlist))]):
        im_id = i * batch_size + im_num 
        objs = [classes[int(x[-1])] for x in output if int(x[0] == im_id)]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:"," ".join(objs)))
        print("-------------------------------------------------------------")
    if CUDA: 
        torch.cuda.synchronize() 
try:
    output 
except NameError:
    print("No Detections were made")
    exit() 

im_dim_list = torch.index_select(im_dim_list,0, output[:,0].long()) 

scaling_factor = torch.min(416/im_dim_list,1)[0].view(-1,1)
output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2



output[:,1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    
    
output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("pallete", "rb"))




draw = time.time()


def write(x, results):
    x = x.cpu().detach().numpy()
    c1 = tuple(x[1:3].astype(int))
    c2 = tuple(x[3:5].astype(int))
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


list(map(lambda x: write(x, loaded_ims), output))

det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))
list(map(cv2.imwrite, det_names, loaded_ims))

end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")


torch.cuda.empty_cache()
