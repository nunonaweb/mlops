# Redução de Dimensionalidade em Imagens para Redes Neurais
# contato@brunoborges.eti.br
# DIO - BairesDev - Machine Learning Practitioner

import requests
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Local Image 
img = cv2.imread('Lenna.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
suave = cv2.GaussianBlur(img, (7, 7), 0)  # aplica blur
(T, bin) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY)
(T, binI) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY_INV)

resultado = np.vstack([
    np.hstack((suave, bin)),
    np.hstack((binI, cv2.bitwise_and(img, img, mask=binI)))
])
cv2.imshow("Binarizacao da imagem", resultado)
cv2.waitKey(0)