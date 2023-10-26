"""
This script read image data from Raspberry Pi Pico board and show the captured images live
The original tutorial (see the link below) uses Processing to read image but Processing does not work well on my Mac
so I developed this script

https://www.arducam.com/docs/pico/arducam-camera-module-for-raspberry-pi-pico/arducam-hm01b0-qvga-camera-module-for-raspberry-pi-pico/
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import serial
import cv2

ser = serial.Serial("/dev/cu.usbmodem1401",9600, timeout = 1)  # you have to change the port here accordingly
frameLength, frameWidth = 64, 64
frameSize = frameLength * frameWidth
identifier = [0x55, 0xAA]


# plot
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

image = np.ones((frameLength, frameWidth), dtype=np.uint8)
ax.imshow(image, cmap='gray', animated=True)

cnt_imageCaptured = 0
cnt_imageSaved = 30

def get_image():

    data = ser.read(int(frameSize * 2.1))
    # ser.reset_input_buffer()
    # ser.flushOutput()
    # while 1:
    #     data = ser.read_until(b'\x55\xAA')
    #     if len(data) == frameSize + 2:
    #         break

    idx = 0
    for idx in range(len(data) - 1):
        if data[idx] == identifier[0] and data[idx + 1] == identifier[1]:  # image starts with this 2byte identifier
            idx += 2  # move over the identifier
            break

    frame = data[idx:idx + frameSize]  # get the image
    frame = np.frombuffer(frame, dtype=np.uint8)  # convert into numpy array
    image = np.reshape(frame, (frameLength, frameWidth))  # reshape
    return image


def updatefig(*args):
    global cnt_imageCaptured, cnt_imageSaved

    image = get_image()

    if cnt_imageCaptured == 0:
        cv2.imwrite("saved_image/szy/sampled/image{}.pgm".format(cnt_imageSaved), image) # save image to local
        print('saved {} images'.format(cnt_imageSaved))
        cnt_imageSaved += 1

    cnt_imageCaptured += 1
    cnt_imageCaptured %= 2

    ax.imshow(image,cmap='gray')
    return ax,

anim = animation.FuncAnimation(fig, updatefig, interval=20, blit=True)
plt.show()







