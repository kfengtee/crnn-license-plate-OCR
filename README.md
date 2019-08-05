# Single-row License Plate Recognition using Convolutional-Recurrent Neural Network (CRNN)

### Usage of Pre-trained License Plate Recognition Model
```ssh
git clone https://github.com/kfengtee/crnn-license-plate-OCR.git
cd crnn-license-plate-OCR
pip install -r requirements.txt
```
```python
import model.alpr as alpr

# create ALPR instance (change parameters according to needs)
lpr = alpr.AutoLPR(decoder='bestPath', normalise=True)

# load model (change parameters according to needs)
lpr.load(crnn_path='model/weights/best-fyp-improved.pth')

# inferencing
lpr.predict('path/to/image')
```

### Test Model Performance on New Dataset
```
python test.py --crnnPath path/to/pretrained/weights --dataPath path/to/test/data --savePath path/to/save/results
```
Optional arguments: <br>
1. **--ctcDecoder** : [*'bestPath'* or *'beamSearch'*], 
    - Method to decode CTC output.
    - Source: https://towardsdatascience.com/beam-search-decoding-in-ctc-trained-neural-networks-5a889a3d85a7
2. **--normalise** : *boolean*,
    - Whether to normalise the posterior probability with prior probability or not (to avoid bias).
    
### Incremental Training / Retrain Model with Own Dataset
```
python train.py --dataPath path/to/training/data --savePath path/to/save/model 
```
To know more about the tunable hyperparameters
```
python ./train.py --help
```
