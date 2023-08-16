import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

# Initialize camera and set its properties
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Initialize SelfieSegmentation and FPS reader
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()

# Load background images
imgDir = "CvProject\images"
if not os.path.exists(imgDir):
    print(f"Error: '{imgDir}' directory does not exist!")
    exit()

imgList = [cv2.imread(os.path.join(imgDir, imgPath)) for imgPath in os.listdir(imgDir) if imgPath.endswith(('.png', '.jpg', '.jpeg'))]

if not imgList:
    print(f"Error: No valid images found in '{imgDir}'!")
    exit()

indexImg = 0

while True:
    success, img = cap.read()
    if not success:
        print("Error: Couldn't read from the camera!")
        break

    imgOut = segmentor.removeBG(img, imgList[indexImg], threshold=0.1)

    imgStack = cvzone.stackImages([img, imgOut], 2, 1)
    _, imgStack = fpsReader.update(imgStack)

    cv2.imshow("Image", imgStack)
    key = cv2.waitKey(1)

    if key == ord('a') and indexImg > 0:
        indexImg -= 1
    elif key == ord('d') and indexImg < len(imgList) - 1:
        indexImg += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
