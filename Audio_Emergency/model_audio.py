#for converting video to audio
import moviepy.editor as mp

# for audio pre-processing and analysis:
import librosa as lib

# for audio features creation
import scipy

# for array processing:
import numpy as np


# for visualizing the data
from matplotlib import pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# to suppress warnings
from warnings import filterwarnings


from tensorflow import keras


# In[2]:

def emergency_audio(clip_east_path,clip_south_path,clip_west_path,clip_north_path,model_path):
    filterwarnings('ignore')
    conv_file = "converted_audio/"
    model = model_path+"CNN_Model_spec.h5"

    clip_east = mp.VideoFileClip(clip_east_path)
    clip_south = mp.VideoFileClip(clip_south_path)
    clip_west = mp.VideoFileClip(clip_west_path)
    clip_north = mp.VideoFileClip(clip_north_path)

    clip_east.audio.write_audiofile(conv_file+"east.wav")
    clip_south.audio.write_audiofile(conv_file+"south.wav")
    clip_west.audio.write_audiofile(conv_file+"west.wav")
    clip_north.audio.write_audiofile(conv_file+"north.wav")



    audio_east,sample_rate = lib.load(conv_file+"east.wav", sr=16000)
    audio_south,sample_rate = lib.load(conv_file+"south.wav", sr=16000)
    audio_west,sample_rate = lib.load(conv_file+"west.wav", sr=16000)
    audio_north,sample_rate = lib.load(conv_file+"north.wav", sr=16000)


    # Data Preparation


    # User-defined Function for Audio chunks audio data is the array

    def audio_chunks(audio_data, num_of_samples=32000, sr=16000):

        # empty list to store new audio chunks formed
        data=[]
        for i in range(0, len(audio_data), sr):

            # creating the audio chunk by starting with the first second & sliding the 2-second window one step at a time
            chunk = audio_data[i: i+ num_of_samples]

            if(len(chunk)==32000):
                data.append(chunk)
        return data



    audio_east=audio_chunks(audio_east)
    audio_south=audio_chunks(audio_south)
    audio_west=audio_chunks(audio_west)
    audio_north=audio_chunks(audio_north)



    #convert data into numpy array to pass through the model
    audio_east=np.array(audio_east)
    audio_south=np.array(audio_south)
    audio_west=np.array(audio_west)
    audio_north=np.array(audio_north)


    #loading the pre trained model
    model = keras.models.load_model(model)

    #applying spectogram features
    def spec_log(audio, sample_rate, eps = 1e-10):

        freq, times, spec = scipy.signal.spectrogram(audio, fs= sample_rate, nperseg=320, noverlap=160)
        return freq, times, np.log(spec.T.astype(np.float32) + eps)

    def extract_spec_features(X_tr):

        # defining empty list to store the features:
        features = []

        # We only need the 3rd array of Spectrogram so assigning the first two arrays as _
        for i in X_tr:
            _,_, spectrogram = spec_log(i, sample_rate)

            mean = np.mean(spectrogram, axis=0)
            std = np.std(spectrogram, axis=0)
            spectrogram = (spectrogram - mean)/std

            features.append(spectrogram)

        # returning the features as array
        return np.array(features)


    # Calling extract function to get training and testing sets:

    audio_east_features = extract_spec_features(audio_east)
    audio_south_features = extract_spec_features(audio_south)
    audio_west_features = extract_spec_features(audio_west)
    audio_north_features = extract_spec_features(audio_north)

    # Computing the probability

    def probability_average(prob,length):
        p=0
        num_chunk=0
        for i in range(length):
            num_chunk +=1
            p = p + float(prob[i][0])
        prob_avg=(p/num_chunk)
        return prob_avg



    # Model Prediction & Computations
    prob_east = model.predict(audio_east_features)
    prob_east_avg = probability_average(prob_east,len(audio_east))
    prob_south = model.predict(audio_south_features)
    prob_south_avg = probability_average(prob_south,len(audio_south))
    prob_west = model.predict(audio_west_features)
    prob_west_avg = probability_average(prob_west,len(audio_west))
    prob_north = model.predict(audio_north_features)
    prob_north_avg = probability_average(prob_north,len(audio_north))


    # Creating a List and Appending to it
    list1=[]

    list1.extend([1-prob_east_avg,1-prob_south_avg,1-prob_west_avg,1-prob_north_avg])


    return list1





