import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import glob



header = st.container()
dataset = st.container()
modelVal = st.container()
model = st.container()

# @st.cache_data
# def load_model(path):
#     model1 = YOLO(path) # "./runs/detect/train/weights/best.pt
#     return model1



with header:
    st.title('AcoVision - YOLOv8n model')
    st.text('AcoVision is a garbage object detection app, adapted to the israeli recycling system.\n'
            'The app is based on the pretrained YOLOv8n model, trained an a custom dataset.')

with dataset:
    st.header('Dataset analysis')

with modelVal:
    st.header('Validation metric')
    mod_val = st.radio("Choose a model:", ('Original', 'Augmented'))

    if mod_val == 'Original':
        # before augmentations (original)
        # confusion_matrix1_path = '/Users/ofir/pythonProject/ITC_Final_Project/yolov8n_EcoVision_Full/val2_conf_0.4_best/confusion_matrix.png'
        confusion_matrix1_path = '/yolov8n_EcoVision_Full/val2_conf_0.4_best/confusion_matrix.png'
        confusion_matrix1 = cv2.imread(confusion_matrix1_path)
        confusion_matrix1 = cv2.cvtColor(confusion_matrix1, cv2.COLOR_BGR2RGB)
        st.subheader('Original dataset')
        st.image(confusion_matrix1)


    # after augmentations
    if mod_val == 'Augmented':
        confusion_matrix2_path = '/Users/ofir/pythonProject/ITC_Final_Project/yolov8n_EcoVision_with_augmentations/runs/detect/yolov8n_EcoVision_fuller3 - best/confusion_matrix.png'
        confusion_matrix2 = cv2.imread(confusion_matrix2_path)
        confusion_matrix2 = cv2.cvtColor(confusion_matrix2, cv2.COLOR_BGR2RGB)
        st.subheader('Augmented dataset')
        st.image(confusion_matrix2)


with model:

    labels = {0: 'paper', 1: 'general', 2: 'packaging', 3: 'glass',
              4: 'cardboard', 5: 'organic', 6: 'deposit bottles', 7: 'electronics',
              8: 'clothing', 9: 'batteries', 10: 'medicines', 11: 'Light bulbs'}
    colors = {0: (0, 0, 255), 1: (209, 200, 200), 2:  (255, 128, 0), 3: (102, 0, 204),
          4: (153, 204, 255), 5: (0, 255, 0), 6: (100, 200, 100), 7: (0, 0, 0),
          8: (255, 51, 153), 9: (204, 204, 0), 10: (153, 153, 255), 11: (255, 255, 0)}

    # image to predict

    images_to_predict_list = glob.glob("/Users/ofir/pythonProject/ITC_Final_Project/images_to_predict/*")
    # print(images_to_predict_list[0])
    st.header('YOLOv8n')
    img_mod, colors_mod1, colors_mod2 = st.columns(3, gap="large")
    image_indx = img_mod.slider('Choose an image as example:', 0, len(images_to_predict_list)-1, 0)
    img1_path = images_to_predict_list[image_indx]
    img1 = cv2.imread(img1_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    img_mod.subheader('Original image')
    img_mod.image(img1)

    colors_mod1.subheader('Classes')
    count = 0
    for indx, label in enumerate(labels.values()):
        count += 1
        new_title = f'<p style="font-family:sans-serif; color:rgb{colors[indx]}; font-size: 22px;"> ------ {label}</p>'
        if count < 7:
            colors_mod1.markdown(new_title, unsafe_allow_html=True)
        else:
            colors_mod2.markdown(new_title, unsafe_allow_html=True)

    choose_mod_col, choose_con_col = st.columns(2)
    mod_ = choose_mod_col.radio("Choose a model", ('Original', 'Augmented'))
    confidence = choose_con_col.slider('Confidence:', 0.0, 1.0, 0.35, 0.05)

    if mod_ == 'Original':
        # import model
        model1_path = "/Users/ofir/pythonProject/ITC_Final_Project/yolov8n_EcoVision_full/runs/detect/train/weights/best.pt"
        model1 = YOLO(model1_path)
        # # predict
        results = model1.predict(source=img1, conf=confidence)
        img = results[0].orig_img
        boxes = results[0].boxes
        fig, ax = plt.subplots()
        for box in boxes:
            box0 = box[0].data[0]
            tx, ty, bx, by = int(box0.data[0]), int(box0.data[1]), int(box0.data[2]), int(box0.data[3])
            box_class = int(box0.data[5])
            prob = float(box0.data[4])
            # st.text(f'{box_class, x, y, w, h}')
            box_cv = cv2.rectangle(img, (tx, ty), (bx, by), colors[box_class], 5)
            cv2.putText(box_cv, f"{labels[box_class]}, {prob:.2f}", (int(tx) + 10, int(ty) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[box_class], 1)
            ax.set_axis_off()
            ax.imshow(box_cv)
        st.subheader('Original model')
        if len(boxes) == 0:
            st.text('No detections')
            st.image(img)
        else:
            st.text(f'{str(len(boxes))} detections')
            st.pyplot(fig)

    if mod_ == 'Augmented':
        # import model
        model2_path = "/Users/ofir/pythonProject/ITC_Final_Project/yolov8n_EcoVision_with_augmentations/runs/detect/train/weights/best.pt"
        model2 = YOLO(model2_path)
        # predict
        results2 = model2.predict(source=img1, conf=confidence)
        img2 = results2[0].orig_img
        boxes2 = results2[0].boxes
        fig2, ax2 = plt.subplots()
        for box_ in boxes2:
            box2 = box_[0].data[0]
            tx2, ty2, bx2, by2 = int(box2.data[0]), int(box2.data[1]), int(box2.data[2]), int(box2.data[3])
            box_class2 = int(box2.data[5])
            prob2 = float(box2.data[4])
            # st.text(f'{box_class, x, y, w, h}')
            box_cv2 = cv2.rectangle(img2, (tx2, ty2), (bx2, by2), colors[box_class2], 5)
            cv2.putText(box_cv2, f"{labels[box_class2]}, {prob2:.2f}", (int(tx2) +10, int(ty2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        colors[box_class2], 1)
            ax2.set_axis_off()
            ax2.imshow(box_cv2)
        st.subheader('Augmented model')
        if len(boxes2) == 0:
            st.text('No detections')
            st.image(img2)
        else:
            st.text(f'{str(len(boxes2))} detections')
            st.pyplot(fig2)