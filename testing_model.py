import numpy as np
import cv2
import pickle

width = 640
height = 480
threshold = 0.75

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

pickle_in = open("trained_model/model_trained.p", "rb")
model = pickle.load(pickle_in)

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

while True:

    success, imgOriginal = cap.read()

    if not success:
        print("Failed to read camera")
        continue

    cv2.rectangle(imgOriginal, (100, 100), (300, 300), (0, 255, 0), 2)

    roi = imgOriginal[100:300, 100:300]

    img = cv2.resize(roi, (32, 32))

    img = preprocessing(img)

    cv2.imshow("Processed Image", img)

    img = img.reshape(1, 32, 32, 1)

    prediction = model.predict(img)

    classIndex = np.argmax(prediction, axis=1)[0]
    probVal = np.max(prediction)

    print("Class:", classIndex)
    print("Probability:", probVal)

    if probVal > threshold:

        cv2.putText(
            imgOriginal,
            str(classIndex) + "  " + str(round(probVal * 100, 2)) + "%",
            (50, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("Original Image", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()