from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable 
import numpy as np
import cv2 

def predict_tranform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride 
    bbox_attrs = 5 + num_classes            # 4 bondingbox + 1 box confidence + classes confidence
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors,grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()             # contiguous() Returns the tensor, order of its element in memory same as input tensor
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # Sigmoid center_x, center_y and objectness score 
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    #Add the Grid offset to center cordinate prediction 
    grid = np.arange(grid_size) 
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda() 
        prediction = prediction.cuda()

    x_y_offset = torch.cat((x_offset, y_offset),1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:, :2] += x_y_offset

    #Apply sigmoid to class score 
    prediction[:,:, 5:5+num_classes] = torch.sigmoid((prediction[:,:,5:5+num_classes]))

    #Resize the detection map to the size of the input image 
    prediction[:,:,:4] *= stride 

    return prediction

def unique(tensor):
    tensor_np = tensor.cpu().numpy() 
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def bbox_iou(box1, box2):
    """
        Returns the bbox iou of two bonding boxes 
    """
    # Get the co-ordinates of bonding boxes 
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3] 
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    #Get the co-ordinates of the intersection rectangle 
    inter_rect_x1 = torch.max(b1_x1, b2_x2)
    inter_rect_y1 = torch.max(b1_y1, b2_y1) 
    inter_rect_x2 = torch.max(b1_x2, b2_x2)
    inter_rect_y2 = torch.max(b1_y2, b2_y2)

    # Intersection area 
    intersection_area = torch.clamp(inter_rect_x2 - inter_rect_x1+1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1+1, min=0)

    #Union area 
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = intersection_area / (b1_area + b2_area - intersection_area)

    return iou

def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    """
    prediction: Prediction values
    confidence: objectness score threshold 
    nms_conf: NMS IOU threshold
    """
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask 

    # Convert (center_x, center_y, height, width) -> (top-left-corner-x, top-left-corner-y, right-bottom-corner-x, right-bottom-corner-y)
    # For calculating iou 
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    """
    A batch of size 3, can have variable no. of true detections
    for eg. image1, image2, image3 has 2,3,5 detection boxes. 
    Therefore confidence and NMS has to be done for one image at once.
    This means we cannot vectorize this operation, must loop over prediction.
    """
    batch_size = prediction.size(0)

    write = False 

    for ind in range(batch_size):
        image_pred = prediction[ind]            # image tensor 
        # Get the class with max score out of 80 classes 
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+num_classes],1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq,1) 

        non_zero_ind = (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue

        # Get the various classes detected in the image 
        img_classes = unique(image_pred_[:, -1]) # -1 index holds the class index 

        for cls in img_classes:
            # Perform NMS 
            #Get the detections with one particular class 
            cls_mask = image_pred_ * (image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

            #sort the detections such that entry with the maximum objectness 
            #Confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)          #no of detection

            for i in range(idx):
                #Get the IOU of all the boxes that come after the one we are looking at 
                #in the loop 
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break 

                except IndexError:
                    break

                #zero out all the detections having IOU > Threshold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask

                #Remove the non-zero entries 
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze() 
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

            batch_ind = image_pred_class.new(image_pred_class.size(0),1).fill_(ind)
            #Repeat the batch id for as many prediction of the class cls in the image
            seq = batch_ind, image_pred_class 
            
            if not write:
                output = torch.cat(seq,1)
                write=True 
            else: 
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    try:
        return output 
    except:
        return 0 



