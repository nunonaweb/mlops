# Redução de Dimensionalidade em Imagens para Redes Neurais
# contato@brunoborges.eti.br
# DIO - BairesDev - Machine Learning Practitioner
import requests
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Download the image (replace with your image URL)
url = 'https://github.com/nunonaweb/mlops/blob/main/Lenna.jpg?raw=true'
response = requests.get(url, stream=True)

# Decode the image data into a NumPy array
img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# Check if image loading was successful
if img is None:
    print("Error: Image could not be loaded. Please check the URL or image format.")
else:
    # Convert to RGB for Matplotlib (optional)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Optional for color accuracy

    # Display the image
    plt.imshow(rgb_img)
    plt.title("Lenna Image"), plt.xticks([]), plt.yticks([])
    plt.show()