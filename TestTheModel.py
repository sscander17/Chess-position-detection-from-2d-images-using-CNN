from keras.models import load_model
import webbrowser
classes = ["black-bishop", "black-king", "black-knight", "black-pawn", "black-queen", "black-rock","empty",
           "white-bishop", "white-king", "white-knight", "white-pawn", "white-queen", "white-rock"]

import pandas as pd
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

model = load_model("C:/Users/sscan/PycharmProjects/EE475Project/chess_best_model.h5")

print(model.summary())

def prepareImage(pathToImage):
    image = load_img(pathToImage)
    imgResult = img_to_array(image)
    imgResult = np.expand_dims(imgResult, axis = 0)
    imgResult = imgResult / 255.
    return imgResult

fen = ""

for i in range(8):
    for j in range(8):
        testImagePath = "C:/Users/sscan/PycharmProjects/EE475Project/squares/"+str(j)+"x"+str(i)+"square.png"
        imageForModel = prepareImage(testImagePath)
        resultArray = model.predict(imageForModel, batch_size=32, verbose=1)
        answer = np.argmax(resultArray, axis=1)
        text = classes[answer[0]]
        print(text)

        match text:
            case "black-bishop":
                fen += "b"
            case "black-king":
                fen += "k"
            case "black-knight":
                fen += "n"
            case "black-pawn":
                fen += "p"
            case "black-queen":
                fen += "q"
            case "black-rock":
                fen += "r"
            case "empty":
                fen += "."
            case "white-bishop":
                fen += "B"
            case "white-king":
                fen += "K"
            case "white-knight":
                fen += "N"
            case "white-pawn":
                fen += "P"
            case "white-queen":
                fen += "Q"
            case "white-rock":
                fen += "R"
            case default:
                fen += "?"
    fen += "/"

print(fen)

fenlist = list(fen)

count = 0
for i in range(72):
    if fenlist[i] == '.':
        count += 1
    else:
        if count >0:
            fenlist[i-1] = count
            count = 0

newfen = ''.join(map(str,fenlist))
print(newfen)
newfen = ''.join(newfen.split('.'))
print(newfen)

url = "lichess.org/analysis/" + newfen
webbrowser.open(url)