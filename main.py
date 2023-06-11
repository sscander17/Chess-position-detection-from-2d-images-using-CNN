# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import cv2
import numpy as np

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def detect_lines(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.imwrite('edges-50-150.jpg', edges)
    minLineLength = 400
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=215, lines=np.array([]),
                            minLineLength=minLineLength, maxLineGap=1500)

    a, b, c = lines.shape
    for i in range(a):
        cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imwrite('houghlines5.jpg', gray)

    return lines


    # # Apply Canny edge detection
    # edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    #
    # # Perform Hough line transformation
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=185)
    #
    # # Draw the detected lines on the original image
    # if lines is not None:
    #     for rho, theta in lines[:, 0]:
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         x1 = int(x0 + 1000 * (-b))
    #         y1 = int(y0 + 1000 * (a))
    #         x2 = int(x0 - 1000 * (-b))
    #         y2 = int(y0 - 1000 * (a))
    #         cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #
    # # Display the result
    # resize = ResizeWithAspectRatio(image, width=600)  # Resize by width OR
    # cv2.imshow('output', resize)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# Path to the chess set image
image_path = 'kontrast.jpeg'

# Detect lines in the image
lines = detect_lines(image_path)
a, b, c = lines.shape


horizontalLines = [[0,0,0,0]]
verticalLines = [[0,0,0,0]]

for i in range(a):
    if abs(lines[i][0][0]-lines[i][0][2]) > abs(lines[i][0][1]-lines[i][0][3]):
        horizontalLines.append([lines[i][0][0], lines[i][0][1], lines[i][0][2], lines[i][0][3]])
    else:
        verticalLines.append([lines[i][0][0], lines[i][0][1], lines[i][0][2], lines[i][0][3]])

horizontalLines.pop(0)
verticalLines.pop(0)

image = cv2.imread(image_path)

# for i in range(len(horizontalLines)):
#     cv2.line(horizontal, (horizontalLines[i][0], horizontalLines[i][1]), (horizontalLines[i][2], horizontalLines[i][3]), (0, 0, 255), 3, cv2.LINE_AA)
#     cv2.imwrite('horizontal.jpg', horizontal)
#
# cv2.imshow('sadf', horizontal)

threshold = 150

verticalPositions = []
for i in range(len(verticalLines)):
    verticalPositions.append((verticalLines[i][0]+verticalLines[i][2])//2)

verticalPositions.sort()

for i in range(len(verticalPositions)-1):
    if verticalPositions[i+1]-verticalPositions[i] < threshold:
        verticalPositions[i+1] = verticalPositions[i]

verticalPositions = list(set(verticalPositions))

horizontalPositions = []
for i in range(len(horizontalLines)):
    horizontalPositions.append((horizontalLines[i][1]+horizontalLines[i][3])//2)

horizontalPositions.sort()

for i in range(len(horizontalPositions)-1):
    if horizontalPositions[i+1]-horizontalPositions[i] < threshold:
        horizontalPositions[i+1] = horizontalPositions[i]

horizontalPositions = list(set(horizontalPositions))

horizontalPositions.sort()
verticalPositions.sort()

# sum = 0
# for i in range(len(horizontalPositions)-1):
#     sum = sum + (horizontalPositions[i+1] - horizontalPositions[i])
#
# slope = sum/(len(horizontalPositions)-1)
#
# horizontalPositions.append(int(horizontalPositions[0]-slope))
#
# horizontalPositions.sort()
# verticalPositions.sort()
#
# verticalPositions[len(verticalPositions)-1] += 10
# verticalPositions[0] += -5
# horizontalPositions[len(verticalPositions)-1] += 30

from PIL import Image

# image = Image.open(image_path)
# for i in range(len(horizontalPositions)-1):
#     for j in range(len(verticalPositions)-1):
#         img2 = image.crop((horizontalPositions[i],verticalPositions[j],horizontalPositions[i+1],verticalPositions[j+1]))
#         img2.save(str(i) + 'x' +str(j) + 'square'+'.png')


image = cv2.imread(image_path)

for i in range(len(verticalPositions)):
    for j in range(len(horizontalPositions)):
        image = cv2.circle(image, (verticalPositions[i],horizontalPositions[j]), radius=15, color=(0, 0, 255), thickness=-1)

# for i in range(len(verticalLines)):
#     cv2.line(vertical, (verticalLines[i][0], verticalLines[i][1]), (verticalLines[i][2], verticalLines[i][3]), (0, 0, 255), 3, cv2.LINE_AA)
#     cv2.imwrite('horizontal.jpg', vertical)
#
#
resized = ResizeWithAspectRatio(image, width=500)  # Resize by width OR
cv2.imshow('sadf', resized)



linesImage = cv2.imread('houghlines5.jpg')
resized = ResizeWithAspectRatio(linesImage, width=600)  # Resize by width OR
cv2.imshow('output', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()


import os
dataDirList = os.listdir("C:/Users/sscan/PycharmProjects/EE475Project/images")

print(dataDirList)

baseDir = "C:/Users/sscan/PycharmProjects/EE475Project"

trainData = os.path.join(baseDir,'train')
os.mkdir(trainData)

validationData = os.path.join(baseDir,'validation')
os.mkdir(validationData)

#train folders
trainWhiteBishopData = os.path.join(trainData, 'white-bishop')
os.mkdir(trainWhiteBishopData)
trainBlackBishopData = os.path.join(trainData, 'black-bishop')
os.mkdir(trainBlackBishopData)

trainWhiteKnightData = os.path.join(trainData, 'white-knight')
os.mkdir(trainWhiteKnightData)
trainBlackKnightData = os.path.join(trainData, 'black-knight')
os.mkdir(trainBlackKnightData)

trainWhitePawnData = os.path.join(trainData, 'white-pawn')
os.mkdir(trainWhitePawnData)
trainBlackPawnData = os.path.join(trainData, 'black-pawn')
os.mkdir(trainBlackPawnData)

trainWhiteKingData = os.path.join(trainData, 'white-king')
os.mkdir(trainWhiteKingData)
trainBlackKingData = os.path.join(trainData, 'black-king')
os.mkdir(trainBlackKingData)

trainWhiteQueenData = os.path.join(trainData, 'white-queen')
os.mkdir(trainWhiteQueenData)
trainBlackQueenData = os.path.join(trainData, 'black-queen')
os.mkdir(trainBlackQueenData)

trainWhiteRockData = os.path.join(trainData, 'white-rock')
os.mkdir(trainWhiteRockData)
trainBlackRockData = os.path.join(trainData, 'black-rock')
os.mkdir(trainBlackRockData)

trainEmptyData = os.path.join(trainData, 'empty')
os.mkdir(trainEmptyData)

#validation folders
valWhiteBishopData = os.path.join(validationData, 'white-bishop')
os.mkdir(valWhiteBishopData)
valBlackBishopData = os.path.join(validationData, 'black-bishop')
os.mkdir(valBlackBishopData)

valWhiteKnightData = os.path.join(validationData, 'white-knight')
os.mkdir(valWhiteKnightData)
valBlackKnightData = os.path.join(validationData, 'black-knight')
os.mkdir(valBlackKnightData)

valWhitePawnData = os.path.join(validationData, 'white-pawn')
os.mkdir(valWhitePawnData)
valBlackPawnData = os.path.join(validationData, 'black-pawn')
os.mkdir(valBlackPawnData)

valWhiteKingData = os.path.join(validationData, 'white-king')
os.mkdir(valWhiteKingData)
valBlackKingData = os.path.join(validationData, 'black-king')
os.mkdir(valBlackKingData)

valWhiteQueenData = os.path.join(validationData, 'white-queen')
os.mkdir(valWhiteQueenData)
valBlackQueenData = os.path.join(validationData, 'black-queen')
os.mkdir(valBlackQueenData)

valWhiteRockData = os.path.join(validationData, 'white-rock')
os.mkdir(valWhiteRockData)
valBlackRockData = os.path.join(validationData, 'black-rock')
os.mkdir(valBlackRockData)

valEmptyData = os.path.join(validationData, 'empty')
os.mkdir(valEmptyData)




