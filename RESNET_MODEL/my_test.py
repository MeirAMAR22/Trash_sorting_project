import streamlit as st
import cv2
import matplotlib.pyplot as plt
import keras
import numpy as np
from PIL import Image
import io

def output_coordinates_to_box_coordinates(cx, cy, w, h, img_h, img_w):
  abs_x = int((cx - w/2) * img_w)
  abs_y = int((cy - h/2) * img_h)
  abs_w = int(w * img_w)
  abs_h = int(h * img_h)
  return abs_x, abs_y, abs_w, abs_h

def my_model():
    model = keras.models.load_model("C:/Users/Meir/Downloads/model_finalyolo77-20230313T214546Z-001/model_finalyolo77")
    return model

def main45(pic, model):
    model = model#my_model()
    print("MODEL LOADED")
    image_path = pic
    c = [
        'battery',
        'biodegradable',
        'book',
        'cardboard',
        'clothes',
        'glass',
        'lunchbox',
        'metal',
        'metals',
        'organics',
        'paper',
        'plastic',
        'wasterpaper',
        'other']

    img = image_path

    # Convert from BGR to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.resize(img, (500, 500))
    net = cv2.dnn.readNetFromDarknet('C:/Users/Meir/Downloads/yolov3.cfg', 'C:/Users/Meir/Downloads/yolov3.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    my_layers = net.getLayerNames()

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (128, 128), swapRB=True, crop=False)
    output_names = net.getUnconnectedOutLayersNames()
    net.setInput(blob)
    large, medium, small = net.forward(output_names)
    all_outputs = np.vstack((small, medium, large))
    objs = all_outputs[all_outputs[:, 4] >= 0.01]
    #print(f'Shape of all_outputs: {all_outputs.shape}')

    with open('C:/Users/Meir/Downloads/coco.names', 'r') as f:
        coco_classes = [line.strip() for line in f.readlines()]

    random_colors = np.random.randint(0, 256, size=(80, 3))

    boxes = []
    confidences = []
    class_names = []
    colors = []
    img_h, img_w = img.shape[:2]

    #print("LET'S DETECT")
    for detection in objs:
        x, y, w, h = output_coordinates_to_box_coordinates(*detection[:4], img_h, img_w)
        boxes.append([x, y, w, h])
        confidences.append(float(detection[4]))
        class_index = np.argmax(detection[5:])

        class_names.append(coco_classes[class_index // 70])
        color = random_colors[np.random.randint(0, 32)].tolist()
        colors.append(color)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.6)
    #print(indices)
    #print(f"I detect {len(indices)} object(s).")

    img_yolo = img.copy()
    my_img = []
    my_res = []
    my_class2 = []

    for i in indices:  # .flatten():
        #print('coucou ', i)
        x, y, w, h = boxes[i]
        class_name = class_names[i]
        confidence = confidences[i]
        color = colors[i]
        text = f'{class_name} {confidence:.3}'

        extracted_img = img[y:y + h, x:x + w]
        #try:
        extracted_img = cv2.resize(extracted_img, (128, 128))
        #except:
        #    pass
        extracted_img = np.array(extracted_img)
        extracted_img = np.expand_dims(extracted_img, axis=0)
        # plt.imshow(extracted_img)
        y_pred = model.predict(extracted_img)
            # print(y_pred)
        text = np.argmax(y_pred)
        #print(text)
        my_class = c[np.argmax(y_pred)]
        my_score = np.max(y_pred) * 100
        plt.imshow(img_yolo)
        plt.show()
        cv2.rectangle(img_yolo, (x, y), (x + w, y + h), color, 5)
        cv2.putText(img_yolo, f'{my_class} - {my_score:.2f}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2)
        #cv2.putText(img_yolo, f'{c[text]} - {np.max(y_pred)*100:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
        plt.axis('off')
        roi = img[y:y + h, x:x + w]
        plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        plt.show;
        my_img.append(roi)
        my_class2.append(my_class)
        my_res.append(my_score)
    return my_img, my_class2, my_res, img_yolo



def main(pic):
    model = my_model()
    print("MODEL LOADED")
    image_path = pic
    c = [
        'battery',
        'biodegradable',
        'book',
        'cardboard',
        'clothes',
        'glass',
        'lunchbox',
        'metal',
        'metals',
        'organics',
        'paper',
        'plastic',
        'wasterpaper',
        'other']
    #image_bytes = image_path.read()
    #img = np.array(Image.open(io.BytesIO(image_bytes)))
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #img2 = cv2.resize(img, (128, 128))
    #img2 = np.expand_dims(img2, axis=0)
    #my_res = model.predict(img2)
    #print('GLOBAL RESULT')
    #print(my_res)

    img = cv2.imread(image_path)

    # Convert from BGR to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (500, 500))
    net = cv2.dnn.readNetFromDarknet('C:/Users/Meir/Downloads/yolov3.cfg', 'C:/Users/Meir/Downloads/yolov3.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    my_layers = net.getLayerNames()

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (128, 128), swapRB=True, crop=False)
    output_names = net.getUnconnectedOutLayersNames()
    net.setInput(blob)
    large, medium, small = net.forward(output_names)
    all_outputs = np.vstack((small, medium, large))
    objs = all_outputs[all_outputs[:, 4] >= 0.01]
    #print(f'Shape of all_outputs: {all_outputs.shape}')

    with open('C:/Users/Meir/Downloads/coco.names', 'r') as f:
        coco_classes = [line.strip() for line in f.readlines()]

    random_colors = np.random.randint(0, 256, size=(80, 3))

    boxes = []
    confidences = []
    class_names = []
    colors = []
    img_h, img_w = img.shape[:2]

    #print("LET'S DETECT")
    for detection in objs:
        x, y, w, h = output_coordinates_to_box_coordinates(*detection[:4], img_h, img_w)
        boxes.append([x, y, w, h])
        confidences.append(float(detection[4]))
        class_index = np.argmax(detection[5:])

        class_names.append(coco_classes[class_index // 70])
        color = random_colors[np.random.randint(0, 32)].tolist()
        colors.append(color)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.6)
    #print(indices)
    #print(f"I detect {len(indices)} object(s).")

    img_yolo = img.copy()

    for i in indices:  # .flatten():
        #print('coucou ', i)
        x, y, w, h = boxes[i]
        class_name = class_names[i]
        confidence = confidences[i]
        color = colors[i]
        text = f'{class_name} {confidence:.3}'

        extracted_img = img[y:y + h, x:x + w]
        extracted_img = cv2.resize(extracted_img, (128,128))
        extracted_img = np.array(extracted_img)
        extracted_img = np.expand_dims(extracted_img, axis=0)
        # plt.imshow(extracted_img)
        y_pred = model.predict(extracted_img)
            # print(y_pred)
        text = np.argmax(y_pred)
        #print(text)
        my_class = c[np.argmax(y_pred)]
        plt.imshow(img_yolo)
        plt.show()
        cv2.rectangle(img_yolo, (x, y), (x + w, y + h), color, 5)
        cv2.putText(img_yolo, f'{my_class}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
        #cv2.putText(img_yolo, f'{c[text]} - {np.max(y_pred)*100:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
        plt.axis('off')
        plt.imshow(cv2.cvtColor(img_yolo, cv2.COLOR_BGR2RGB))
        plt.show();

        #print(my_class)
    return img_yolo


if __name__ == '__main__':
    main("C:/Users/Meir/Downloads/vase2.jpeg")
