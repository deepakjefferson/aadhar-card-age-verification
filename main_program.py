from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time   
import numpy as np
import cv2
import glob
from imutils import paths
import os
import os.path
import datetime
AadharCard={}
cap=cv2.VideoCapture(0)
i=0
j=8
k=0
while True:
    while k<8:
        ret,frame=cap.read()
        cv2.putText(frame,'image will be captured ' +str(j)+'in seconds',(40,475), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame,(140,410),(360,380), (0, 0, 200), 2)
        cv2.putText(frame,'place your card',(40,80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.imshow("frame",frame)
        key=cv2.waitKey(1)
        k=k+1
    k=0
    i=i+1
    j=8-i
    if i>8:
        ret,frame=cap.read()
        cv2.imwrite('photo.jpg',frame)
        img=cv2.imread("photo.jpg")
        # crop = img[130:370, 370:420]
        crop = img[360:420, 140:380]  
        cv2.imwrite('photo.jpg',crop)
        cap.release()
        cv2.destroyAllWindows
        # k=1500
        break
    
    
cap.release()
cv2.destroyAllWindows



import pytesseract
from PIL import Image

aadharNumberImage= Image.open('photo.jpg')
aadharNumberImage= aadharNumberImage.convert('L')
aadharNumberImage= aadharNumberImage.point(lambda x: 0 if x < 200 else 255)
aadharNumberSplitString= pytesseract.image_to_string(aadharNumberImage)
x = aadharNumberSplitString.split()
joinedAadharNumber=x[0]+x[1]+x[2]
print(joinedAadharNumber)
print(len(joinedAadharNumber))

driver = webdriver.Chrome()
driver.get('https://myaadhaar.uidai.gov.in/')
driver.refresh()
driver.refresh()


def loginButton():
    driver.find_element(By.XPATH, '/html/body/div[1]/div/section/div/div[2]/div/button').click()

def enterCredentials():
    enterAadhaar=driver.find_element(By.XPATH,'/html/body/main/div/form/div[1]/label/input')
    # AadharNumber='484591589292'
    #AadharNumber= input('ENTER YOUR NUMBER: ')
    AadharNumber=joinedAadharNumber
    enterAadhaar.send_keys(AadharNumber)
    time.sleep(2)
    captcha=captchaRecognizer()
    print(captcha)
    # captcha= input('ENTER CAPTCHA: ')
    EnterAboveCaptcha=driver.find_element(By.XPATH,'/html/body/main/div/form/div[3]/label/input')
    # print(EnterAboveCaptcha)
    EnterAboveCaptcha.send_keys(captcha)
    driver.find_element(By.XPATH,'/html/body/main/div/form/button[1]').click()
    otp=input('enter otp: ')
    EnterOtp=driver.find_element(By.XPATH,'/html/body/main/div/form/div[5]/label/input')
    EnterOtp.send_keys(otp)
    driver.find_element(By.XPATH,'/html/body/main/div/form/button[2]').click()

def captchaRecognizer():
    driver.find_element(By.XPATH,'//*[@id="captcha_block"]/img').screenshot('captcha.png')
    # captcha extraction starts
    unsolvedCapchasFolder='rac'
    outputFolder='extracted'
    unsolvedCapchas=glob.glob(os.path.join(unsolvedCapchasFolder,'*'))
    counts={}
    for captcha_image in unsolvedCapchas:
    
        # grab the base filename as the text
        filename=os.path.basename(captcha_image)
        captchaText=os.path.splitext(filename)[0]
    
        # Load the image and convert it to grayscale
        image = cv2.imread(captcha_image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     
        # Adding some extra padding around the image
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    
        # applying threshold
        thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1]
    
        # creating empty list for holding the coordinates of the letters
        letter_image_regions = []
         
        # finding the contours
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # now we will loop through each of the contours and extract the letter
        for contour in contours:
          # Get the rectangle that contains the contour
          (x, y, w, h) = cv2.boundingRect(contour)
                
          # checking if any counter is too wide
          # if countour is too wide then there could be two letters joined together or are very close to each other
          if w / h > 1.25:
            # Split it in half into two letter regions
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
          else:  
            letter_image_regions.append((x, y, w, h))
        
        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
        
        # Save each letter as a single image
        for letter_bounding_box, letter_text in zip(letter_image_regions,captchaText):
          # Grab the coordinates of the letter in the image
          x, y, w, h = letter_bounding_box
        
          # Extract the letter from the original image with a 2-pixel margin around the edge
          letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]
        
          # Get the folder to save the image in
          save_path = os.path.join(outputFolder, letter_text)
        
          # creating different output folder for storing different letters
          if not os.path.exists(save_path):
            os.makedirs(save_path)
        
          #   write the letter image to a file
          count = counts.get(letter_text, 1)
          p = os.path.join(save_path, "{}.png".format(str(count)))
          cv2.imwrite(p, letter_image)
        
          # increment the count
          counts[letter_text] = count + 1
    
    letter_folder = 'extracted'
    
    #creating empty lists for storing image data and labels
    data = []
    labels = []
    for image in paths.list_images(letter_folder):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (30,30))
        # adding a 3rd dimension to the image
        img = np.expand_dims(img, axis = 2)
        # print(img)
        #grabing the name of the letter based on the folder it is present in
        label = image.split(os.path.sep)[-2]
        
        # appending to the empty lists
        data.append(img)
        labels.append(label)
    
    #converting data and labels to np array
    data = np.array(data, dtype = "float")
    labels = np.array(labels)
    data=data/255.0
    
    # split the dataset into train and tests set
    from sklearn.model_selection import train_test_split
    (train_x,val_x,train_y,val_y)=train_test_split(data,labels,test_size=0.2,random_state=0)
    
    # one hot encoding our target variable 'labels
    from sklearn.preprocessing import LabelBinarizer
    import pickle
    lb=LabelBinarizer().fit(train_y)
    train_y=lb.transform(train_y)
    val_y=lb.transform(val_y)
    
    bin=pickle.dumps(lb)
    with open("capcha_labels.pickle",'wb')as f:
       pickle.dump(lb,f)
    
    
    from keras.models import Sequential
    from keras.layers.convolutional import Conv2D, MaxPooling2D
    from keras.layers.core import Flatten, Dense, Dropout
    from keras.callbacks import EarlyStopping
    
    # #building model
    model = Sequential()
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=(30, 30, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(61, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    
    # using early stoping for avoiding overfitting
    estop = EarlyStopping(patience=10, mode='min', min_delta=0.001, monitor='val_loss')
    model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=32, epochs=50, verbose=1, callbacks = [estop])
    
    # Load the image and convert it to grayscale
    image = cv2.imread('captcha.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Add some extra padding around the image
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    
    # threshold the image
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1]
    
    # find the contours
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
        
    letter_image_regions = []
    
    # Now we can loop through each of the contours and extract the letter
    
    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        
        # checking if any counter is too wide
        # if countour is too wide then there could be two letters joined together or are very close to each other
        if w / h > 1.25:
            # Split it in half into two letter regions
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            letter_image_regions.append((x, y, w, h))
                
    
    # Sort the detected letter images based on the x coordinate to make sure
    # we get them from left-to-right so that we match the right image with the right letter  
    
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    
    # Create an output image and a list to hold our predicted letters
    output = cv2.merge([gray] * 3)
    predictions = []
        
    # Creating an empty list for storing predicted letters
    predictions = []
        
    # Save out each letter as a single image
    for letter_bounding_box in letter_image_regions:
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box
    
        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]
    
        letter_image = cv2.resize(letter_image, (30,30))
            
        # Turn the single image into a 4d list of images
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)
    
        # making prediction
        pred = model.predict(letter_image)
            
        # Convert the one-hot-encoded prediction back to a normal letter
        letter = lb.inverse_transform(pred)[0]
        predictions.append(letter)
    
    
        # draw the prediction on the output image
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    
    # Print the captcha's text
    captcha_text = "".join(predictions)
    # print("CAPTCHA text is: {}".format(captcha_text))
    
    # Show the annotated image
    # cv2.imshow('output',output)
    # cv2.waitKey(0)

    return(captcha_text)

def myAadhar():
   time.sleep(5)
   soup=BeautifulSoup(driver.page_source,'lxml')
#    print(soup)
   ExtractedDetails1=soup.find("div", {'class':'name-english'})
   ExtractedDetails2=soup.find("div", {'class':'dob'})
   ExtractedDetails3=soup.find("div", {'class':'gender'})
   ExtractedDetails4=soup.find("div", {'class':'aadhaar-back__address-english'})
#    print(ExtractedDetails1)
#    print(ExtractedDetails2)
#    print(ExtractedDetails3)
#    print(ExtractedDetails4)
   AadharCard={
    'NAME':ExtractedDetails1.text,
    'DOB':ExtractedDetails2.text,
    'GENDER':ExtractedDetails3.text,
    'ADDRESS':ExtractedDetails4.text
   }
   print(AadharCard)
   a=AadharCard['DOB']
 #  print(a)
   DOBsplit= a.split(' ')
   DOB=DOBsplit[1]
 #  print(DOB)
   DOBsplit=DOB.split('/')
 #  print(DOBsplit)
   age=0
   DOBday=int(DOBsplit[0])
   DOBmonth=int(DOBsplit[1])
   DOByear=int(DOBsplit[2])
  # print(DOBday,DOBmonth,DOByear)
   todayDate= datetime.datetime.now()
   currentMonth=int(todayDate.strftime("%m"))
   currentDay=int(todayDate.strftime("%d"))
   currentYear=int(todayDate.strftime("%Y"))
   if ((currentMonth==DOBmonth)and(currentDay>=DOBday))or((currentMonth>DOBmonth)):
       age=currentYear-DOByear
   else:
       age=currentYear-DOByear-1
   print('Hiii !!!',AadharCard['NAME'],'\n','Your current age is ',age)


def faceRecognition():
   aadharPhotoDownload()
   photoCapture()
   faceComparision()
      
def aadharPhotoDownload():
   time.sleep(5)
   soup2=BeautifulSoup(driver.page_source,'html.parser')
   images=soup2.find_all('img')
   photo=images[6]
   src=photo['src']
   driver.get(src)
          
def photoCapture():
    cap=cv2.VideoCapture(0)
    i=0
    j=4
    k=0
    while True:
        while k<8:
            ret,frame=cap.read()
            cv2.putText(frame,'image will be captured ' +str(j)+'in seconds',(40,460), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame,(490,100),(150,380), (0, 0, 200), 4)
            cv2.putText(frame,'place your face inside the square',(40,80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.imshow("frame",frame)
            key=cv2.waitKey(1)
            k=k+1
        k=0
        i=i+1
        j=4-i
        if i>4:
            cv2.imwrite('photo.jpg',frame)
            img=cv2.imread("photo.jpg")
            crop = img[100:380, 150:490]  
            cv2.imwrite('photo.jpg',crop)
            break
    cap.release()
    cv2.destroyAllWindows   

def faceComparision():
   import face_recognition
   img=cv2.imread('photo.jpg')
   rgb_img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
   img_encoding=face_recognition.face_encodings(rgb_img)[0]
   img2=cv2.imread('download.jpeg')
   rgb_img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
   img_encoding2=face_recognition.face_encodings(rgb_img2)[0]
   result= face_recognition.compare_faces([img_encoding],img_encoding2)
   print('result: ',result)
   if result==[True]:
       print('your face matches')
   else:
       print('your face does not match')



loginButton()
enterCredentials()
myAadhar()
photoCapture()
faceRecognition()

while True:
    i=1



