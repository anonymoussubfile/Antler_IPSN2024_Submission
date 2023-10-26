"""
This script read image data from Raspberry Pi Pico board and show the image
The original tutorial (see the link below) uses Processing to read image but Processing does not work well on my Mac
so I developed this script

https://www.arducam.com/docs/pico/arducam-camera-module-for-raspberry-pi-pico/arducam-hm01b0-qvga-camera-module-for-raspberry-pi-pico/
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import serial

ser = serial.Serial("/dev/cu.usbmodem101",9600, timeout = 1) # you have to change the port here accordingly
frameLength, frameWidth = 64, 64
frameSize = frameLength * frameWidth

def get_image():

    data = ser.read(int(frameSize * 3))
    idx = 0
    for idx in range(len(data) - 1):
        if data[idx] == 0x55 and data[idx + 1] == 0xAA:  # image starts with this 2byte identifier
            idx += 2  # move over the identifier
            break

    frame = data[idx:idx + frameSize]  # get the image
    frame = np.frombuffer(frame, dtype=np.uint8)  # convert into numpy array
    image = np.reshape(frame, (frameLength, frameWidth))  # reshape
    return image

# plot
fig = plt.figure()


image = get_image()
plt.imshow(image, cmap='gray')
plt.show()





