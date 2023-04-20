"""
Sound_test

Test for picking up audio from microphone

Plus some signal processing

Date-Created: 2022 NOV 29
Date-Last-Modified: 2023 APR 17
Author: Netp
"""

import pyaudio
import struct
import keyboard
import numpy as np
import matplotlib.pyplot as plot
import sys
import os
import pydirectinput as dir
from typing import List, Dict

scale : Dict[int, str] = {
    466 : "A#4",
    523 : "C5",
    587 : "D5",
    622 : "D#5",
    698 : "F5",
    783 : "G5",
    830 : "G#5",
    880 : "A5",
    932 : "A#5",
    1046: "C6",
    1108: "D#6",
    1174: "D6",
    1244: "D#6",
    1864: "A#6",
}

# Mouse bindings
bindings_m = {
    # Down
    "D#5" : lambda: dir.moveRel(0, MOUSE_SPD, relative=True, _pause=False),
    # Up
    "D6" : lambda: dir.moveRel(0, -MOUSE_SPD, relative=True, _pause=False),
    # Left
    "G5" : lambda: dir.moveRel(0 - MOUSE_SPD, 0, relative=True, _pause=False),
    # Right
    "A#5" : lambda: dir.moveRel(MOUSE_SPD, 0, relative=True, _pause=False),
    # Hold left click
    "C#4" : lambda: dir.mouseUp(button="left"),
    "B3" : lambda: dir.mouseDown(button="left"),
    # Single clicks
    "A#6" : dir.leftClick,
    "G4" : dir.rightClick,
    "F#4" : dir.middleClick,
}

def hz_to_note(freq : float):
    freq = int(freq)
    keys_within_range = [key for key in scale.keys() if abs(key - freq) <= N2F_RANGE]
    if keys_within_range != []:
        note = scale[keys_within_range[0]]
        if note in bindings_m.keys():
            bindings_m[note]()

def on_close(event):
    global running
    running = False
    fig.canvas.flush_events()
    stream.stop_stream()
    stream.close()
    p.terminate()
    plot.close(fig)

def display_freq(freqs : List[float]):
    os.system("cls")
    for counter, f in enumerate(freqs):
        print(f"Frequency {counter + 1}: {round(f,2)}Hz")
    hz_to_note(freqs[0])

CHUNK = 1024 * 2
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
# RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "output.wav"

# FFT constants
N2F_RANGE = 20
GRAINS = 10000
# Convert to close enough frequencies
# Definitely a math reason for this, I'm just stupid
FFT2F_CONSTANT = 0.817

# Control constants
MOUSE_SPD = 12

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

fig, (ax, ax2) = plot.subplots(2)

x_fft = np.linspace(0, RATE, CHUNK)
x = np.arange(0, 2 * CHUNK, 2)

line = ax.plot(x, np.random.rand(CHUNK), 'r')[0]
line_fft = ax2.semilogx(x_fft, np.random.rand(CHUNK))[0]

ax.set_ylim(-60000, 60000)
ax.set_xlim(0, CHUNK)

ax2.set_ylim(0, 1)
ax2.set_xlim(20, RATE/2)

fig.canvas.mpl_connect('close_event', on_close)

fig.show()

while running:
    data = stream.read(CHUNK)
    dataInt = struct.unpack(str(CHUNK) + 'h', data)

    line.set_ydata(dataInt)

    line_fft.set_ydata(np.abs(np.fft.fft(dataInt)) * 2 / (33000 * CHUNK))

    # Peak frequency calculation
    N = len(dataInt) # Number of samples
    # xfft = np.fft.fft(dataInt) # Compute FFT (1000 default)
    xfft = np.fft.fft(dataInt, GRAINS)
    xfft = np.abs(xfft[:N//2]) # Keep only positive frequencies
    # Find the frequency bin with the highest magnitude
    peak_bin = np.argmax(xfft)
    # Convert the frequency bin to a frequency in Hz
    peak_frequency = (peak_bin * RATE / N) * FFT2F_CONSTANT

    display_freq([peak_frequency])

    fig.canvas.draw()
    fig.canvas.flush_events()

print("* done recording")
sys.exit(0) # 0 means no errors