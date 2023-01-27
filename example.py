
import numpy as np
import matplotlib.pyplot as plt

from main import *

## PREPARE AUDIO ##

# Start by creating an instance of ProcessAudio to prepare recording for further processing &
# generate window functions
example_audio = ProcessAudio("recordings/download.wav", 261.6)

# Change duration, show info
example_audio.set_duration(0.6,1.4)
data = example_audio.samples
print(example_audio)

# Playback to ensure correct
Audio(data=example_audio.samples, rate= example_audio.rate)

# Input notes, M and partials, K that we are interested in 
example_audio.generate_M_K_w([261.6,440], [10,15])
#generates w with 10 partials on 261.6hz and 15 on 440 

# Generate D matrix representation using window function and M, K, w values
D = example_audio.generate_D()

## ANALYSE AUDIO ##
piano_sample = AnalyseAudio(D, data)

# Calculate maximum likelihood B
piano_sample.MLE()

# Resynthisize sound using D @ B and listen
ml_estimation = piano_sample.resynth_sound()
Audio(data=ml_estimation, rate= example_audio.rate)

