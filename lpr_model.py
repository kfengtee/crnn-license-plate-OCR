from __future__ import division
from __future__ import print_function

# self-defined functions
import crnn_model
import utils 

import torch
from torch.autograd import Variable
from PIL import Image
import string
import torchvision.transforms as transforms
import numpy as np

# import the necessary packages for EAST 
from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
from scipy.special import softmax

def crnn_predict(crnn, img, transformer):
    """
    Returns
    ------
    out: a list of tuples (predicted alphanumeric sequence, confidence level)
    """
    output = []
    
    classes = string.ascii_uppercase + string.digits
    image = img.copy()
    
    image = transformer(image)
    if torch.cuda.is_available():
        image.cuda()
    image = image.view(1, *image.size())
    
    # forward pass (convert to numpy array)
    preds_np = crnn(image).data.cpu().numpy().squeeze()
    
    # move first column to last (so that we can use CTCDecoder as it is)
    preds_np = np.hstack([preds_np[:, 1:], preds_np[:, [0]]])
    
    preds_sm = softmax(preds_np, axis=1)
    
#     output = ctcBestPath(preds_sm, classes)
    output = utils.ctcBeamSearch(preds_sm, classes, None)
        
    return output

class EAST_CRNN:
    
    def __init__(self):
        
        # crnn parameters
        self.IMGH = 32
        self.nc = 1 
        alphabet = string.ascii_uppercase + string.digits
        self.nclass = len(alphabet) + 1
        self.transformer = transforms.Compose([
            transforms.Grayscale(),  
            transforms.Resize((32,100)),
            transforms.ToTensor()])
        
        # east parameters
        self.layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        self.min_conf = 0.95
        self.img_width = 320
        self.img_height = 160
        
        # prediction parameters
#         self.num_top_results = 5

        # assumption threshold
        self.crop_ratio_thre = 0.2
        self.intersect_ratio_thre = 0.4
        self.extra_crop = 0.255
        
    def load(self, east_path, crnn_path):
        # load EAST
        self.east = cv2.dnn.readNet(east_path)

        # load CRNN
        self.crnn = crnn_model.CRNN(self.IMGH, self.nc, self.nclass, nh=256)
        self.crnn = torch.nn.DataParallel(self.crnn, range(1))
        
        if torch.cuda.is_available():
            self.crnn = self.crnn.cuda()
            self.crnn.load_state_dict(torch.load(crnn_path))
        else:
            self.crnn.load_state_dict(torch.load(crnn_path, map_location='cpu'))
        
        self.crnn.eval()  # remember to set to test mode (otherwise some layers might behave differently)
        
    def predict(self, img_path):
        
        # load the input image and grab the image dimensions
        self.image = cv2.imread(img_path)
        
        if self.image is None:
            raise Exception('Invalid image path: {}'.format(img_path))
        # orig = image.copy()
        (H, W) = self.image.shape[:2]

        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (newW, newH) = (self.img_width, self.img_height)
        rW = W / float(newW)
        rH = H / float(newH)

        # resize the image and grab the new image dimensions
        self.image = cv2.resize(self.image, (newW, newH))
        (H, W) = self.image.shape[:2]
        
        blob = cv2.dnn.blobFromImage(self.image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        
        
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
        self.image = Image.open(img_path).convert('L')

        ## (crop_size, startX, startY, endX, endY)
        self.crop_info = []

        # loop over the bounding boxes and collect cropping info
        for (startX, startY, endX, endY) in boxes:

            # scale the bounding box coordinates based on the respective ratios
            startX = int(startX * rW) 
            startY = int(startY * rH) 
            endX = int(endX * rW) 
            endY = int(endY * rH) 

            # with additional noise (extra crop)
            noiseX = (endX - startX) * self.extra_crop
            noiseY = (endY - startY) * self.extra_crop

            startX_new = int(startX - noiseX)
            startY_new = int(startY - noiseY)
            endX_new = int(endX + noiseX)
            endY_new = int(endY + noiseY)

            crop_size = (endX_new - startX_new) * (endY_new - startY_new)

            self.crop_info.append({"crop_size":crop_size, "crop_coord":(startX_new, startY_new, endX_new, endY_new)})

        # find the largest crop size 
        if len(self.crop_info) != 0:
            max_crop_size = max([x["crop_size"] for x in self.crop_info])

        # FILTER 1: Remove very small crops (most probably brand name)
        for i, single_crop_info in enumerate(self.crop_info):
            # avoid cropping characters under the license plate (if any), like brands / dealer info
            if (single_crop_info["crop_size"] / max_crop_size) <= self.crop_ratio_thre:
                #remove detected text region that is too small
                self.crop_info.pop(i)
        
        # FILTER 2: Remove overlapping crops (might result in repeated predictions)
        if len(self.crop_info) == 2:
            startX1, startY1, endX1, endY1 = self.crop_info[0]["crop_coord"]
            startX2, startY2, endX2, endY2 = self.crop_info[1]["crop_coord"]

            # find all the intersection points
            x_intersect = np.intersect1d(np.arange(startX1, endX1), np.arange(startX2, endX2))
            y_intersect = np.intersect1d(np.arange(startY1, endY1), np.arange(startY2, endY2))
            
            # if intersection exists
            if len(x_intersect) >= 2 and len(y_intersect) >= 2:
                area_intersect = np.diff(x_intersect[[0, -1]]) * np.diff(y_intersect[[0, -1]])
                area1 = self.crop_info[0]["crop_size"]
                area2 = self.crop_info[1]["crop_size"]
                
                # BAD PRACTICE: HARD CODING
                if area_intersect / area1 > self.intersect_ratio_thre:
                    self.crop_info.pop(0)
                elif area_intersect / area2 > self.intersect_ratio_thre:
                    self.crop_info.pop(1)
        
        # We assume if there's 2 detected text (After FILTER1 and 2) => multi line license plate        
        if len(self.crop_info) == 2:       
            pred_conf_multi = []
            for single_crop_info in self.crop_info:
                cropped = self.image.crop(single_crop_info["crop_coord"])
                pred_conf_multi.append(crnn_predict(self.crnn, cropped, self.transformer))


            #### find a combinations using matrix (linear algebra) => argmax => indices ###

            top_pred1 = pred_conf_multi[0][0][0]
            top_pred2 = pred_conf_multi[1][0][0]
            pct_char1 = sum(c.isalpha() for c in top_pred1) / len(top_pred1)
            pct_char2 = sum(c.isalpha() for c in top_pred2) / len(top_pred2)

            # first prediction is the numeric part, swap position first
            if pct_char1 < pct_char2:
                pred_conf_multi[0], pred_conf_multi[1] = pred_conf_multi[1], pred_conf_multi[0]


#             alpha_conf = np.mat([x[1] for x in pred_conf_multi[0]])
#             numeric_conf = np.mat([x[1] for x in pred_conf_multi[1]])

#             cross_table = np.matmul(alpha_conf.T, numeric_conf)

#             pred_conf = []
#             for top_n_results in range(self.num_top_results):
#                 idx = utils.max_idx(cross_table)[0]
#                 pred_conf.append((pred_conf_multi[0][idx[0]][0] + pred_conf_multi[1][idx[1]][0], cross_table[idx]))
#                 cross_table[idx[0]] = np.NINF

            return "".join(pred_conf_multi)
        # if detected text regions are not exactly 2, assume it's single line with noises
        else:
            pred = crnn_predict(self.crnn, self.image, self.transformer)
            return pred
    
    def cropped_image(self):
        for single_crop_info in self.crop_info:
            cropped = self.image.crop(single_crop_info["crop_coord"])
            cropped.show()
