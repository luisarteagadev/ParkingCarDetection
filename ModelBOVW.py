import cv2 as cv
import numpy as np
from joblib import load


class BOVW:
    @staticmethod
    def predict_image(model_path, dictionary_path, image_path, class_names):
        try:
            model = load(model_path)
        except Exception as e:
            raise ValueError(f"No se pudo cargar el modelo: {model_path}. Error: {e}")

        try:
            kmeans = BOVW.load_kmeans(dictionary_path)
        except Exception as e:
            raise ValueError(f"No se pudo cargar el diccionario: {dictionary_path}. Error: {e}")

        histogram = BOVW.preprocess_image(image_path, kmeans)
        prediction = model.predict(histogram)
        predicted_class = class_names[prediction[0]]
        return predicted_class

    @staticmethod
    def preprocess_image(I, kmeans):
        #img = cv.imread(I, cv.IMREAD_GRAYSCALE)
        if I is None:
            raise ValueError(f"No se pudo transformar la imagen a escala de grises: ")

        sift = cv.SIFT_create()
        _, descriptors = sift.detectAndCompute(I, None)

        if descriptors is not None:
            words = kmeans.predict(descriptors)
            histogram, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1))
            histogram = histogram.astype('float32')
            histogram /= histogram.sum()
        else:
            histogram = np.zeros(kmeans.n_clusters, dtype='float32')

        return histogram.reshape(1, -1)

    @staticmethod
    def load_kmeans(dictionary_path):
        try:
            kmeans = load(dictionary_path)
            return kmeans
        except Exception as e:
            raise ValueError(f"Error al cargar el modelo KMeans: {e}")
