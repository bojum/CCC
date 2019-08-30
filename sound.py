## extracts features from audio files and converts into numpy
import librosa
import numpy as np
import os

audio_dir = '/Users/panchanok/Desktop/PyHack2019/sound/tone_perfect/'
audio_files = os.listdir(audio_dir)

#audio_path = '/Users/panchanok/Desktop/PyHack2019/sound/tone_perfect/a1_FV1_MP3.mp3'

## return a one-D array with mcff, mcff_delta, and chomagram of the audio file
def vectorize(audio_path):


    y, sr = librosa.load(audio_path)

    # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
    hop_length = 32

    # Separate harmonics and percussives into two waveforms
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Beat track on the percussive signal
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
                                                 sr=sr)

    # Compute MFCC features from the raw signal
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

    # And the first-order differences (delta features)
    mfcc_delta = librosa.feature.delta(mfcc)

    # # Stack and synchronize between beat events
    # # This time, we'll use the mean value (default) instead of median
    # beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]),
    #                                     beat_frames)

    # Compute chroma features from the harmonic signal
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
                                            sr=sr)

    # # Aggregate chroma features between beat events
    # # We'll use the median value of each feature between beat frames
    # beat_chroma = librosa.util.sync(chromagram,
    #                                 beat_frames,
    #                                 aggregate=np.median)

    # # Finally, stack all beat-synchronous features together
    # beat_features = np.vstack([beat_chroma, beat_mfcc_delta])
    # #tempo, neat_frames = librosa.beat.beat_track(y=y, sr=sr)

    #processed_input = np.vstack([beat_features, beat_chroma, beat_mfcc_delta])


    return np.hstack([mfcc.flatten(), mfcc_delta.flatten(), chromagram.flatten()])

### vecterize all audio files
input = np.array([vectorize(audio_dir + f) for f in audio_files[1:5]])
print(input[1][4])
### Export list of arrays
with open('______ss.csv', 'w') as f:
    for item in input:
        f.write("%s\n" % item)

print('succeed')
