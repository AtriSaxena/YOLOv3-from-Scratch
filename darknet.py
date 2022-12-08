from __future__ import division 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable 
import numpy as np 
import cv2 
import sys 
from utils.util import predict_tranform



def get_test_input():
    img = cv2.imread('test_image/dog-cycle-car.png') 
    img = cv2.resize(img, (416,416))
    img_ = img[:,:,::-1].transpose((2,0,1)) #BGR 2 RGB | H,X,W,C -> C,X,H,X,W 
    img_ = img_[np.newaxis,:,:,:]/255.0     # Add a channel 0 for batch
    img_ = torch.from_numpy(img_).float() 
    img_ = Variable(img_)
    return img_

def parse_cfg(cfgfile):
    """
    Takes a config file 

    Returns a list of blocks. Each blocks describe a block in the neural network
    to be built. Block is represented as a dictionary in the list.
    """

    file = open(cfgfile,'r')
    lines = file.read().split('\n')                # store the lines in list
    lines = [x for x in lines if len(x) > 0]       # get rid of empty lines 
    lines = [x for x in lines if x[0]!= '#']       # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]   # get rid of whitespaces

    block = {} 
    blocks = [] 

    for line in lines: 
        if line[0] == "[":                      # This mark the start of the new block 
            if len(block) != 0:                # If block is not empty, implies it is storing values of previous block
                blocks.append(block)            # Add it to the block list 
                block = {}                      # re-init the block
            block["type"] = line[1:-1].rstrip() 
        else: 
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip() 
    blocks.append(block)
    return blocks 

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def create_modules(blocks):
    net_info = blocks[0]                          # Captures details of input and preocessing
    module_list = nn.ModuleList()                 # Like a normal list containing nn.Module objects
    prev_filters = 3                              # Keep track of No of filters in previous layers
    output_filters = []                           # Keep track of No of output filters in all blocks for Routing layers

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential() 

        if x['type'] == "convolutional":
            #Get info about layer

            activation = x['activation'] 
            try:
                batch_normalize = x['batch_normalize']
                bias = False 
            except:
                batch_normalize =0 
                bias = True 
            filters = int(x['filters'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            padding = int(x['pad'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            
            # Add the convolutional layer 
            conv = nn.Conv2d(prev_filters, filters,
                                kernel_size, stride, 
                                pad, bias= bias)
            module.add_module(f"conv_{index}", conv)

            # Add the batch norm layer

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f"batch_norm_{index}", bn)

            #Check the activation
            # It is either Linear or Leaky Relu for YOLO 
            if activation == "leaky":
                activation_layer = nn.LeakyReLU(0.1, inplace=True)
                module.add_module(f"leaky_{index}", activation_layer)
        elif x['type'] == "upsample":
            stride = x['stride']
            upsample = nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=True)
            module.add_module(f"upsample_{index}", upsample)

        elif x['type'] == "route":
            x["layers"] = x["layers"].split(",")

            #start of route 
            start = int(x["layers"][0])

            # end of route if exist 
            try:
                end = int(x["layers"][1])
            except:
                end = 0 
            # Positive annotation 
            if start > 0: 
                start = start - index 
            if end > 0: 
                end = end - index 
            route = EmptyLayer()
            module.add_module(f"route_{index}", route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index+end]
            else:
                filters = output_filters[index + start]

        elif x["type"] == "shortcut":
            shortcut = EmptyLayer() 
            module.add_module(f"shortcut_{index}", shortcut)

        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module(f"Detection_{index}",detection)
        
        module_list.append(module)
        prev_filters = filters 
        output_filters.append(filters)

    return (net_info, module_list)

# blocks = parse_cfg("cfg/yolov3.cfg")
# print(create_modules(blocks))


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__() 
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}                        # Cache the output for route layers
        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x) 
            elif module_type == "route": 
                layers = module["layers"] 
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] -1
                
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if (layers[1]) > 0: 
                        layers[1] = layers[1] - i 

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]] 

                    x = torch.cat((map1, map2),1)
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]

            elif module_type == "yolo": 
                anchors = self.module_list[i][0].anchors 
                #Get the input dimensions 
                inp_dim = int(self.net_info["height"])

                #Get the number of classes 
                num_classes = int(module["classes"]) 

                #Transform 
                x = x.data 
                x = predict_tranform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x 
                    write = 1 
                else: 
                    detections = torch.cat((detections, x),1)
        
            outputs[i] = x
    
        return detections





            


model = Darknet("cfg/yolov3.cfg")
inp = get_test_input() 
pred = model(inp, torch.cuda.is_available())
print(pred)











