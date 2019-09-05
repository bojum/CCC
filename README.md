# Closer
The program determines the closest Chinese consonant to what user speaker. This will help user knows if their pronunciation is off to how a native will say.

The code is implemented in Jupyter Notebook.

## record_sound.ipynb
The notebook makes use of sounddevice library to record user's pronunciation thru the built-in microphone. The audio file is saved for later process.

## process_sound.ipynb
It reads in audio files and processes them into MCFF and Chromatogram. Each audio file is modified into the same format so that they are all compatible to the model from model.ipynb. 

## model.ipynb
This notebook initiates all necessary ingredients for the model. Train and validation calls occur in this file.

## *_attr_in.csv
Processed audio file that is ready to be fed into the model. It is in the same order as in *_target_in.csv.

## *_target_in.csv
Labels of each audio file used for training/validating. It is in the same order as in *_attr_in.csv.
