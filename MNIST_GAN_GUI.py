from MNIST_GAN import gen, NOISE_SHAPE, saveImage
from tkinter import *
import numpy as np
import tensorflow
import png
import os
import atexit


class App(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()

    def mainloop(self):
        super().mainloop()


def deleteTempImage():
    os.remove(os.path.join(".", "temp.png"))


atexit.register(deleteTempImage)

master = App()

prod = 1
for i in NOISE_SHAPE:
    prod *= i

sliders = []

for i in range(prod):
    w = Scale(master, from_=-3, to=3, orient=HORIZONTAL, resolution=0.01)
    sliders.append(w)
    w.grid(row=i % 10, column=i // 10)

saveImage(np.zeros(shape=(28, 28)), name="temp", grayscale=True)

image = PhotoImage(
    file=os.path.join(".", "temp.png"))

label = Label(image=image)
label.image = image  # keep a reference!
label.pack()

while True:

    values = np.reshape(np.array([i.get()
                                  for i in sliders]), (1,) + NOISE_SHAPE)

    new_image_raw = gen(values)[0]

    saveImage(new_image_raw, name="temp", grayscale=True)

    new_image = PhotoImage(
        file="temp.png")

    new_image = new_image.zoom(10, 10)

    label.configure(image=new_image)
    label.image = new_image

    master.update()
