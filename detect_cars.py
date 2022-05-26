import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip
from timeit import default_timer as timer
import sys

weights_path = os.path.join("yolo", "yolov3.weights")
config_path = os.path.join("yolo", "yolov3.cfg")
labels_path = os.path.join("yolo", "coco.names")

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
names = net.getLayerNames()
# (h, w) = img.shape[:2]
(h, w) = (720, 1280)
layers_names = [names[int(i)-1] for i in net.getUnconnectedOutLayers()]
labels = open(labels_path).read().strip().split("\n")


def yolo_pipeline(img):

    blob = cv2.dnn.blobFromImage(
        img, 1/255.0, (416, 416), crop=False, swapRB=False)
    net.setInput(blob)
    # start_t = time.time()
    layers_output = net.forward(layers_names)
    # print("A forword pass through yolov3 took{}".format(time.time()-start_t))

    boxes = []
    confidences = []
    classIDs = []
    for output in layers_output:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if(confidence > 0.85):
                box = detection[:4] * np.array([w, h, w, h])
                bx, by, bw, bh = box.astype("int")

                x = int(bx - (bw / 2))
                y = int(by - (bh / 2))

                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.8)
    if len(idxs) == 0:
        return img
    for i in idxs.flatten():
        # for i in range(len(boxes)):
        (x, y) = [boxes[i][0], boxes[i][1]]
        (W, H) = [boxes[i][2], boxes[i][3]]
        cv2.rectangle(img, (x, y), (x + W, y + H), (0, 255, 255), 2)
        cv2.putText(img, f'{labels[classIDs[i]]} {round(confidences[i],3)}',
                    (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 139, 139), 2)

    return img


def create_output(input_path):

    clip_ = VideoFileClip(input_path)
    output_path = f'{input_path.split(".")[0]}_car_detection.mp4'
    clip = clip_.fl_image(yolo_pipeline)
    clip.write_videofile(output_path, audio=False)
    print(f'output saved to >> {output_path}')


if __name__ == "__main__":

    input_path = sys.argv[1]

    # start = timer()
    create_output(input_path)
    # print("with GPU:", timer()-start)
