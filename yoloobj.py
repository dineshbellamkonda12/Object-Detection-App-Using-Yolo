import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # using multiple cameras make it as 1
width = 320
confThreshold = 0.5
nmsThreshold = 0.3

# will create while loop so that we can get the frames of our webcam

# need to load yolo 3 model, since it was trained on coco dataset,first collect
# the names of our classes, We can use coco names file to actually extract all these
# class names
classesFile = 'coco.names'  # This is a text file
classNames = []

# open this text file and extract all the information
with open(classesFile, 'rt') as f:  # read the text file as 'rt'
    classNames = f.read().rstrip('\n').split('\n')  # it extracts all the info based on new line
# print(classNames) # We have 80 different names, all of them are stored in 'classNames'
# print(len(classNames))

# We are going to find our configuration file that has the architecture details
# and then we are going to find the weights file that has all the trained weights
# inside of it, To do that we can go to the yolo website pjreddie.com/darknet/yolo/

# We need to import configuration file and weight file so that we can create our network
modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

# Now we are going to create our network

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
# Now we are going to use open cv as back end
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    print(indices)
    for i in indices:
        # i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', (x,y-10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255),2)


while True:
    success, img = cap.read()  # This will gives us our image and it will tell us weather it was successful to
    # retrieve or not
    # The next step is actually input our image to the network, We cannot just input the image
    # plain image that we are getting from our webcam in to our network
    # The network only accepts particular type of format called "blob"
    # so we have to convert this image to blob
    # Here width and height are same, so we have width, width
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (width, width), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()  # This will give all the names of our layers
    # print(layerNames)
    # print(net.getUnconnectedOutLayers())
    # We have to extract only the output layers
    outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]  # To get the names of output layers
    # print(outputNames)
    # Now we can send this image as a forward pass to our network, and we can find output of the layers
    outputs = net.forward(outputNames)
    # print(len(outputs))
    # lets open this output up and see what is inside,Where are the bounding boxes, Where are the object names & ids
    # or other information like confidence level & stuff
    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    # print(outputs[0][0])

    findObjects(outputs, img)
    # This is a matrix that has 300 rows & 85 different columns
    cv2.imshow('Image', img)  # Image as our window name and img that we want to display
    cv2.waitKey(1)  # Delay it for 1 milli second
