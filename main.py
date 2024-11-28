
import sys

sys.path.append("D:\\Programacion\\DesarrolloPython\\DetectaClasifica\\yolov7")

import torch
import cv2 as cv
import joblib

from models.experimental import attempt_load
from torchvision import transforms
from PIL import Image
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device, time_synchronized
from time import  sleep
from ModelBOVW import BOVW



### Cargando modelos
model = attempt_load('bestPesos.pt', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model.eval()

#svm_model = joblib.load('D:\Programacion\DesarrolloPython\DetectaClasifica\svm_model.pkl')
model_path = "D:\Programacion\DesarrolloPython\DetectaClasifica\svm_model.pkl"
dictionary_path = "D:\Programacion\DesarrolloPython\DetectaClasifica\kmeans_dictionary.pkl"
image_path = "ruta_a_la_imagen.jpg"
class_names = ['Sedan', 'Pickup', 'Suv']



video_path = 'D:/Programacion/DesarrolloPython/DetectaClasifica/video_parking.mp4'
cap = cv.VideoCapture(video_path)

class_colors = {
    'Sedan': (0, 255, 255),   # Amarillo para "Sedan"
    'Pickup': (0, 165, 255),  # Naranja para "Pickup"
    'Suv': (128, 0, 128)      # Púrpura para "Suv"
}


FPS = int(cap.get(cv.CAP_PROP_FPS))

contador=1
img_size = check_img_size(640, s=model.stride.max())
device = select_device('')
while cap.isOpened():
    val,frame=cap.read()
    if not val:
        break
    #print(type(frame))
    #print(frame.shape)

    ###PREPROCESAMIENTO
    imagen_original=np.copy(frame)
    imagen_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    resized_image = cv.resize(frame, (640, 448))
    transposed_frame = np.transpose(resized_image, (2, 0, 1))###(de HxWxC a CxHxW)
    img = torch.from_numpy(transposed_frame).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    ###INFERENCIA
    pred = model(img)[0]
    # Aplicar NMS (supresión de no-máximos)
    pred = non_max_suppression(pred, 0.25, 0.45, agnostic=True)
    for det in pred:  # det: Detecciones de la imagen
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()  # Escalar las coordenadas
            for *xyxy, conf, cls in det: # coordenadas, confianza, clase
                label = f'{model.names[int(cls)]} {conf:.2f}'  # Nombre de la clase y la confianza

                # CLASIFICACION Y VISUALIZACION DE RESULTADOS
                if int(cls) == 0:  # Clase 0: puedes cambiar a la clase que desees
                    color = (0, 0, 255)  # Color rojo para clase 1
                    label = f'Car: {conf:.2f}'

                    car_detected = imagen_gray[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    dim=car_detected.shape
                    if(dim[0] >0 and dim[1]>0):
                        predicted_class = BOVW.predict_image(model_path,dictionary_path,car_detected,class_names)
                    # output_dir = 'D:\Programacion\DesarrolloPython\DetectaClasifica\images_cars'
                    #filename = os.path.join(output_dir, f'car_{counter}.jpg')
                    #cv.imwrite(filename, cropped_image)
                    #counter += 1
                else:
                    color = (0, 255, 0)  # Color azul para otras clases
                    label = f'Free: {conf:.2f}'
                # Dibuja las cajas de la detección en la imagen
                cv.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
                cv.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if int(cls) == 0:
                    color_for_class = class_colors.get(predicted_class, (255, 255, 255))

                    # Desplazar la posición del texto para que no se solape con el label
                    y_offset = int(xyxy[1]) - 30  # Puedes ajustar la posición según sea necesario
                    cv.putText(frame, predicted_class, (int(xyxy[0]), y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.5,  color_for_class, 2)
    cv.imshow('Clasificacion en estacionamiento', frame)
    sleep(1/FPS)
    if cv.waitKey(1)  == 27:
        break
cap.release()
cv.destroyAllWindows()
