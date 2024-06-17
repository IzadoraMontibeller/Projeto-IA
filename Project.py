from keras.models import load_model
import numpy as np
import cv2
import time

model = load_model('Keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

cap = cv2.VideoCapture(0)

classes = ['JEISSON', 'MATEUS', 'LEANDRA', 'NAO IDENTIFICADO!!']

start_time = None

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (224, 224))
    image_array = np.asarray(imgS)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    indexVal = np.argmax(prediction)

    cv2.putText(img, str(classes[indexVal]), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
#    print(classes[indexVal])

    cv2.imshow('img', img)
    cv2.waitKey(1)

    if classes[indexVal] != 'NAO IDENTIFICADO!!':
        if start_time is None:
            start_time = time.time()
        else:
            elapsed_time = time.time() - start_time
            if elapsed_time >= 5:
                print("Pessoa Identificada: ", classes[indexVal])
                break
    else:
        start_time = None

cap.release()
cv2.destroyAllWindows()