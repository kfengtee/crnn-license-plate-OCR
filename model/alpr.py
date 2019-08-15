from __future__ import division
from __future__ import print_function

# self-defined functions
import model.crnn as crnn
import utils 

import torch
from torch.autograd import Variable
from PIL import Image
import string
import torchvision.transforms as transforms
import numpy as np
from scipy.special import softmax

# text detection
from imutils.object_detection import non_max_suppression
import cv2

import pickle
with open('model/weights/prior.pkl', 'rb') as f:
    prior = pickle.load(f)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def crnn_predict(crnn, img, transformer, decoder='bestPath', normalise=False):
    """
    Params
    ------
    crnn: torch.nn
        Neural network architecture
    transformer: torchvision.transform
        Image transformer
    decoder: string, 'bestPath' or 'beamSearch'
        CTC decoder method.
    
    Returns
    ------
    out: a list of tuples (predicted alphanumeric sequence, confidence level)
    """
    
    classes = string.ascii_uppercase + string.digits
    image = img.copy()
    
    image = transformer(image).to(device)
    image = image.view(1, *image.size())
    
    # forward pass (convert to numpy array)
    preds_np = crnn(image).data.cpu().numpy().squeeze()
    
    # move first column to last (so that we can use CTCDecoder as it is)
    preds_np = np.hstack([preds_np[:, 1:], preds_np[:, [0]]])
    
    preds_sm = softmax(preds_np, axis=1)
#     preds_sm = np.divide(preds_sm, prior)
    
    # normalise is only suitable for best path
    if normalise == True:
        preds_sm = np.divide(preds_sm, prior)
            
    if decoder == 'bestPath':
        output = utils.ctcBestPath(preds_sm, classes)
        
    elif decoder == 'beamSearch':
        output = utils.ctcBeamSearch(preds_sm, classes, None)
    else:
        raise Exception("Invalid decoder method. \
                        Choose either 'bestPath' or 'beamSearch'")
        
    return output

class AutoLPR:
    
    def __init__(self, decoder='bestPath', normalise=False):
        
        # crnn parameters
        self.IMGH = 32
        self.nc = 1 
        alphabet = string.ascii_uppercase + string.digits
        self.nclass = len(alphabet) + 1
        self.transformer = transforms.Compose([
            transforms.Grayscale(),  
            transforms.Resize(self.IMGH),
            transforms.ToTensor()])
        self.decoder = decoder
        self.normalise = normalise
        
        # east parameters
        self.layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        self.min_conf = 0.5
                
    def load(self, crnn_path, east_path='model/weights/frozen_east_text_detection.pb'):
        
        # load EAST
        self.east_path = east_path
#         self.east = cv2.dnn.readNet(east_path)
        
        # load CRNN
        self.crnn = crnn.CRNN(self.IMGH, self.nc, self.nclass, nh=256).to(device)
        self.crnn.load_state_dict(torch.load(crnn_path, map_location=device))
            
        # remember to set to test mode (otherwise some layers might behave differently)
        self.crnn.eval()
        
    def predict(self, img_path):
        
        image = cv2.imread(img_path)
        
        (H, W) = image.shape[:2]
        (newH, newW) = (round(H / 32) * 32, (round(W / 32)) * 32) # scale to multiple of 32
        
        newH = newH if newH > 0 else 32
        newW = newW if newW > 0 else 32
        
        rW = W / float(newW)
        rH = H / float(newH)
        
        image = cv2.resize(image, (newW, newH))
        
        (H, W) = image.shape[:2]
        
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), 
                                     swapRB=True, crop=False)
        
        #### text detection using EAST ####
        self.east = cv2.dnn.readNet(self.east_path)
        self.east.setInput(blob)
        (scores, geometry) = self.east.forward(self.layerNames)
        
        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
        
        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < self.min_conf:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        
        
        # image processing for crnn
        image = Image.open(img_path)
        
        if len(boxes) == 2:
            # determine vertical or horizontal plate
            # if we take the difference between both coordinates,
            # vertical plate have large difference io y coordinates diff[0, 2]
            # horizontal plate have large difference on x coordinates diff[1,3]
            diff = np.absolute(np.array(boxes[0]) - np.array(boxes[1]))
            if (diff[0] + diff[2]) < (diff[1] + diff[3]):
                # sort boxes from top to bottom (normal way of reading LP)
                boxes = sorted(boxes, key=lambda x: x[1], reverse=False)
            else:
                boxes = sorted(boxes, key=lambda x: x[0], reverse=False)
            
            pred = []
            for (startX, startY, endX, endY) in boxes:
                startX = int(startX * rW) 
                startY = int(startY * rH) 
                endX = int(endX * rW) 
                endY = int(endY * rH) 
                
                cropped = image.crop((startX, startY, endX, endY))
                
                pred.append(crnn_predict(self.crnn, cropped, self.transformer, 
                                         self.decoder, self.normalise))
            return "".join(pred)
        else:
            return crnn_predict(self.crnn, image, self.transformer, 
                                self.decoder, self.normalise)