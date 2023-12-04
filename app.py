
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_image_select import image_select
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import time

def run():
    
    st.title("Vérsejt számláló")
    st.text("Megadja, hogy adott képen hány darab vérsejt látható")
        
    option = st.selectbox(
    'Kép feltöltése vagy alapértelmezett képek közül választás:',
    ('Példa képek', 'Feltöltés'))

    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    
    if option == 'Feltöltés':
        data_file = st.file_uploader("Kép feltöltése")

        if data_file is not None:
            image_array = Image.open(data_file)
            st.image(image_array)
        else:
            image_array = Image.open("SIMCEPImages_A01_C1_F1_s01_w1.TIF")
        with st.spinner('Kép feldolgozása...'):
            time.sleep(3)
        st.success('Done!')

        
    else:

        test_id_list = ["SIMCEPImages_A01_C1_F1_s01_w1.TIF", "SIMCEPImages_A11_C44_F1_s21_w1.TIF", "SIMCEPImages_A19_C78_F1_s25_w2.TIF"]

        cv2_image_jpg_list = []
        for i in range(len(test_id_list)):
            img = cv2.imread(test_id_list[i])
            img_resized = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
            cv2.imwrite(f'image_{i}.jpg', img_resized)
            cv2_image_jpg_list.append(f'image_{i}.jpg')
        

        img = image_select("Label", cv2_image_jpg_list)
        st.write(img)
                
        index_of_selected_image = cv2_image_jpg_list.index(img)
        st.write("A kiválasztott kép sorszáma: ", index_of_selected_image+1)
        
        with st.spinner('Kép feldolgozása...'):
            time.sleep(3)
        st.success('Done!')

        image_array = Image.open(test_id_list[index_of_selected_image])
        
    if image_array is not None:
            
        img = image_array.save("img.jpg")
        img = cv2.imread("img.jpg")
        
    else: 
        image_array = Image.open(test_id_list[0])

    model = tf.keras.models.load_model('CellCounterModel.h5')
    
    IMG_CHANNELS = 3
    X_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)) #, dtype=np.bool)
    
    if option == 'Feltöltés':

        cv2_image = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        cv2.imwrite('mask.jpg', cv2_image)
        mask = cv2.imread('mask.jpg')
        
        X_test[0] = mask
        
        preds = model.predict(X_test)
        
        result = ' Pred: ' + str(int(preds))
        plt.title(result, fontsize=14)
        plt.axis('off')
        plt.savefig(f'tmp_test.png')
        st.image('tmp_test.png')
        
    else:

        path_mask = './BBBC005_v1_ground_truth/' + test_id_list[index_of_selected_image]

        cv2_image = cv2.imread(path_mask)
        cv2_image = cv2.resize(cv2_image, (IMG_HEIGHT, IMG_WIDTH))
        cv2.imwrite('mask.jpg', cv2_image)
        mask = cv2.imread('mask.jpg')
    
        X_test[0] = mask

        preds = model.predict(X_test)
        
        test_image = X_test[0]
        result = ' Pred: ' + str(int(preds))
        plt.title(result, fontsize=14)
        plt.axis('off')
        plt.savefig(f'tmp_test.png')

        st.image('tmp_test.png')
        
        

run()