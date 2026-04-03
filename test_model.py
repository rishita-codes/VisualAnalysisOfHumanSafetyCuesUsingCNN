from tensorflow.keras.models import load_model

model= load_model("/content/drive/MyDrive/DL_ML_Models/mobilenetv2_model.keras")

from google.colab import files

uploaded= files.upload()
img_path = list(uploaded.keys())[0]

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_image(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.axis('off')
plt.title("Uploaded Image")
plt.show()

img_processed = preprocess_image(img_path)

preds = model.predict(img_processed)

class_id = np.argmax(preds)

labels= ["aggressive", "distress", "fallen", "normal"]
print("Predicted class:", class_id)
print("Prediction:", labels[class_id])