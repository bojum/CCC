## extracts features from audio files and converts into numpy
import librosa
import numpy as np
import os, re, csv
from datetime import datetime

global hop_length


# Set the hop length; at 22050 Hz, 512 samples ~= 23ms
hop_length = 128

##save the current directory
cwd = os.getcwd()

##change to sound file directory !!hard-coded
audio_dir = '/Users/panchanok/Desktop/PyHack2019/sound/tone_perfect/'
os.chdir(audio_dir)

##list files in the directory
audio_files = os.listdir(audio_dir)[1:10]
#print(audio_files)


## return a (flatten) one-D array of mfcc of an audio file
def getMFCC(audio_file):

    print('*getting ', audio_file)
    y, sr = librosa.load(audio_file)
    # Compute MFCC features from the raw signal
    return librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13).flatten()

## return a (flatten) one-D array of chromagram of an audio file
def getChroma(audio_file):

    y, sr = librosa.load(audio_file)
    # Separate harmonics and percussives into two waveforms
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Compute chroma features from the harmonic signal
    return librosa.feature.chroma_cqt(y=y_harmonic,
                                            sr=sr).flatten()


## return a list of 1-d array of MFCC padded with 0's of ALL audio files
def getPaddedMFCC(audio_files):
    result = [getMFCC(f) for f in audio_files]

    ##pad arrays with 0's. Get arrays of size Max
    max_len = max([len(x) for x in result])
    padded = [np.pad(x, (0, max_len - len(x)), mode = 'constant') for x in result]

    ##sanity check
    is_shorter = sum([len(x) - max_len for x in padded])
    if is_shorter < 0:
        print('not padded well')
        return -1
    else:
        return padded

## return a list of 1-d array of chromagram padded with 0's of ALL audio files
def getPaddedChroma(audio_files):
    result = [getChroma(f) for f in audio_files]

    ##pad arrays with 0's. Get arrays of size Max
    max_len = max([len(x) for x in result])
    padded = [np.pad(x, (0, max_len - len(x)), mode = 'constant') for x in result]

    ##sanity check
    is_shorter = sum([len(x) - max_len for x in padded])
    if is_shorter < 0:
        print('not padded well')
        return -1
    else:
        return padded

print('getting mfcc')
mfcc = getPaddedMFCC(audio_files)
print('getting chromagram')
chrom = getPaddedChroma(audio_files)

if mfcc == -1 or chrom == -1:
    print('ATTENTION: some instance is not padded')
    exit()

## concatenate mfcc and chrom features
attr_input = [np.hstack([m, c]) for m, c in zip(mfcc, chrom)]

## checking the final length
#print(len(mfcc[5]), len(chrom[5]), len(x[5]))


##detect targets from sound names
p = re.compile('^[aeou]|[bcdfghjklmnpqrstwxyz]+(?=[aeiou])')
target_input = [p.match(f).group() for f in audio_files]

##for debugging
# for i in range(len(audio_files)):
#     f = audio_files[i]
#     print(f)
#     p.match(f).group()

## check corrrectness
#print(target)
#print(audio_files)


# ## tag labels to attributes. Return 2-d array.
# labeled_input = np.array([np.hstack([i, l]) for i, l in zip(att_input, target)])


os.chdir(cwd)
this_time = datetime.now().strftime('%H_%M_%S')
attr_export_name = 'attr_in_' + this_time + '.csv'
target_export_name = 'target_in_' + this_time + '.csv'

with open(attr_export_name,"w+") as processed:
    csvWriter = csv.writer(processed,delimiter=',')
    csvWriter.writerows(attr_input)

with open(target_export_name,"w+") as processed:
    csvWriter = csv.writer(processed,delimiter=',')
    csvWriter.writerows(target_input)

## check final lengths
#print(len(att_input[3]), len(labeled_input[3]), labeled_input)
print('Attribute data saved as ', attr_export_name)
print('Target data saved as ', target_export_name)
