import cv2
import numpy as np


def getImageStream():
    cam = cv2.VideoCapture(0)
    result, img = cam.read()

    if not result:
        print("Error finding webcam")
        return

    scale = 0.1
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    resized = cv2.resize(img, dim)

    # upsized = cv2.resize(resized, (int(img.shape[1]), int(img.shape[0])))
    # cv2.imshow("Webcam Picture", upsized)
    # cv2.waitKey(0)
    # cv2.destroyWindow("Webcam Picture")
    print("resized size:", resized.shape)

    serialised = np.ravel(resized)
    serialised_bits = np.ravel([np.unpackbits(x) for x in serialised])
    return serialised_bits


def getImage(bitStream, imgSize):
    chunks = np.split(bitStream, bitStream.size//8)
    chunks_int = np.array([int(np.packbits(i)) for i in chunks], np.uint8)
    img_int = np.reshape(chunks_int, imgSize)

    scale = 10
    width = int(imgSize[1] * scale)
    height = int(imgSize[0] * scale)
    dim = (width, height)
    print("dim", dim)
    print("img shape:", img_int.shape)
    print("img dtype:", img_int.dtype)
    upsized_img = cv2.resize(img_int, dim)

    cv2.imshow("Received picture", upsized_img)
    cv2.waitKey(0)
    cv2.destroyWindow("Received picture")
    return upsized_img


if __name__ == "__main__":
    bitStream = getImageStream()
    print("bitStream:", bitStream)
    imgSize = (48, 64, 3)
    imgOriginal = getImage(bitStream, imgSize)