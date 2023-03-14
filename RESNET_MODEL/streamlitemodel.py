import streamlit as st
import cv2
import matplotlib.pyplot as plt
import keras
import numpy as np
import my_test
import os
import io


def main2():

    model = keras.models.load_model("C:/Users/Meir/Downloads/model_finalyolo77-20230313T214546Z-001/model_finalyolo77")
    # Define the page title
    iii = cv2.imread("C:/Users/Meir/Downloads/ISRAELGARB.jpg")
    iii = cv2.cvtColor(iii, cv2.COLOR_BGR2RGB)
    #imgj = cv2.imdecode(np.fromstring(iii.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(iii)#imgj)
    #st.title("EcoVision")
    st.markdown("<h1 style='text-align: center; color: red;'>EcoVision</h1>", unsafe_allow_html=True)

    image_path = st.file_uploader("Select a picture to help your planet breath (without any pressure of course)", type=["jpg", "jpeg", "png"])
    img = 0
    if image_path is not None:
        st.write("Image path :", image_path)
        img = cv2.imdecode(np.fromstring(image_path.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(img)

    my_img, my_class, my_res, img_final = my_test.main45(img, model)
    st.title('PREDICTED RESULTS')
    st.title(f'{len(my_img)} objects were found.')
    #st.image(img_final)
    st.image(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB))
    st.title('Display RESULTS')
    #st.image(my_test.main(f'C:/Users/Meir/Downloads/{image_path.name}'))

    for i in range(len(my_img)):
        col1, col2, col3 = st.columns(3)
        #taux = my_img[i].shape[0]/2
        #iimmgg = cv2.resize(my_img[i], (round(my_img[i].shape[0]/taux), round(my_img[i].shape[1]/taux)))
        iimmgg = my_img[i]
        with col1:
            st.header(f"{my_class[i]}")
            st.image(iimmgg, use_column_width=True)

        iii3 = cv2.imread("C:/Users/Meir/Downloads/recyclearrow.png")
        with col2:
            st.image(iii3, use_column_width=True)

        iii4 = cv2.imread("C:/Users/Meir/Downloads/poubelle_verte.jpeg")
        with col3:
            st.header(f"Confidence: {my_res[i]:.2f}%")

            st.image(iii4, use_column_width=True)
        #st.image(my_test.main45(img))



if __name__ == '__main__':
    main2()