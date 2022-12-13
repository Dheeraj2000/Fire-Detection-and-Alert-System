import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
import smtplib
import pyttsx3  # pip install pyttsx3 # to convert text to speech

#Load the saved model
model = tf.keras.models.load_model('InceptionV3.h5')
video = cv2.VideoCapture(0)

engine = pyttsx3.init()

def speak(audio):
    engine.say(audio)
    engine.runAndWait()  # wait until function is completed


def sendEmail(to, content):
    server = smtplib.SMTP('smtp.gmail.com', 587) #mailing service to use
    server.ehlo()  #help in identifing ourselves to smtp server
    server.starttls()   # which will helps us in putting connection to the smtp server into the TLS model
    # for this function, you must enable low security in your gmail which you are going to use as sender

    server.login('dheerajrohilla2020@gmail.com', 'pass')
    try:
        server.sendmail('dheerajrohilla2020@gmail.com', to, content)
        print("Email sent")
    except:
        print('error sending notification')
    #server.quit()
    server.close()



a = 1

while True:
        _, frame = video.read()
#Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')
#Resizing into 224x224 because we trained the model with this image size.
        im = im.resize((224,224))
        img_array = image.img_to_array(im)
        img_array = np.expand_dims(img_array, axis=0) / 255
        probabilities = model.predict(img_array)[0]
        #Calling the predict method on model to predict 'fire' on the image
        prediction = np.argmax(probabilities)
        #if prediction is 0, which means there is fire in the frame.
        if prediction == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            #print(f"'\033[91m' Fire detected with probability {probabilities[prediction]}")
            print(f"Fire detected with probability {probabilities[prediction]}")
            cv2.putText(frame, "Fire detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
            if probabilities[prediction] > 0.65:
                if a == 1:
                    sendEmail('dheerajrohilla2000@gmail.com', 'fire detected take action')
                    a = a + 1
                speak("Fire Detected")

        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
            break
video.release()
cv2.destroyAllWindows()
