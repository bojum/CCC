[Please see CNN-mfcc-9 branch]

# CCC (Chinese Consonant Classifier)
The program determines the closest Chinese consonant to what the user speaks. The classification results will help improve the pronunciation so that they can pronounce as close to native speakers as possible. 

The code is implemented in Jupyter Notebook.

## How to Run

1. Clone the repository to your local machine (make sure you clone CNN-mfcc-9 branch).
There are two important directories in the repo. The first one is temp/ directory which is where your record will be saved. The other one is model8/ directory which is where .pt file of trained data lives.

2. Open jupyter notebook 'record_sound.ipynb' -> click run all <br/> 
When you run it, one of the code chunks will indicate that you'll have 3 seconds before the record begins. Then, you'll have 1.25 seconds to record your voice. The audio file will be automatically saved in temp/ directory. You can listen to your record on the lower code chunk. The wave plot is also provided.

3. Open jupyter notebook 'model_CNN_classify.ipynb' -> click run all <br/> 
This notebook will load the trained model from model8/ directory into the current network and feed your record that was saved ealier into the model. On the last line is a Chinese consonant the model thinks it is closest to your record. The line says: "It thinks you said: \[your classified sound\]"
        

## Files
### record_sound.ipynb
The notebook makes use of sounddevice library to record user's pronunciation thru the built-in microphone. The audio file is saved for later process.

### process_sound_MFCC.ipynb
It reads in audio files and processes them into MCFF and Chromatogram. Each audio file is modified into the same format so that they are all compatible to the model from model.ipynb. 

### model_CNN_train.ipynb

### model_CNN_classify.ipynb

