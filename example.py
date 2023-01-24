
import numpy as np
import matplotlib.pyplot as plt

from main import *


# Start by creating an instance of ProcessAudio to prepare recording for further processing &
# generate window functions
example_audio = ProcessAudio("recordings/download.wav", 261.6)

# Change duration, show info
example_audio.set_duration(0.6,1.4)
print(example_audio)

# Playback to ensure correct
Audio(data=example_audio.samples, rate= example_audio.sampling_frequency)