import model.alpr as alpr
import argparse
import os
import pandas as pd
import editdistance
from sklearn.metrics import accuracy_score

#### argument parsing ####
parser = argparse.ArgumentParser()
parser.add_argument('--dataPath', required=True, help='path to training dataset')
parser.add_argument('--savePath', required=True, help='path to save results')
parser.add_argument('--crnnPath', required=True, help='path to pre-trained CRNN model')
parser.add_argument('--ctcDecoder', type=str, default='bestPath', 
                    choices=['bestPath', 'beamSearch'],
                    help='method for decoding ctc outputs')

parser.add_argument('--normalise', type=bool, default=False, 
                    help='set true to normalise posterior probability.')

opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.savePath):
    os.makedirs(opt.savePath)


#### load model ####
lpr = alpr.AutoLPR(decoder=opt.ctcDecoder, normalise=opt.normalise)
lpr.load(crnn_path=opt.crnnPath)


#### test performance ####
result = pd.DataFrame([], columns=['path', 'gTruth', 'pred', 'editDistance'])

for file in os.listdir(opt.dataPath):
    if file != '.ipynb_checkpoints':
        
        # ground truth
        filename, file_extension = os.path.splitext(file)
        gt = filename.split('_')[-1]
        
        # prediction
        pred = lpr.predict(os.path.join(opt.dataPath, file))
        
        # distance
        dist = editdistance.eval(gt, pred)
        
        result = result.append({'path': file,
                                'gTruth': gt,
                                'pred': pred,
                                'editDistance': dist}, ignore_index=True)

        
#### print and save results ####
print("Accuracy:", accuracy_score(result.gTruth, result.pred))
print('\n')
print("Edit Distance Distribution")
print(result.editDistance.value_counts(sort=False))

result = result.sort_values("editDistance", ascending=False).reset_index(drop=True)
result.to_csv(os.path.join(opt.savePath, 'result.csv'),
              index=False)