"""
Tool used to detect faces in real time using the method proposed by Paul Viola and Michael Jones in
their paper, "Rapid Object Detection using a Boosted Cascade of Simple Features" (2001).
"""
import argparse
import cv2
import logging
import numpy as np
import os
import sys

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def capture_webcam_img():
    """Capture a single image with the webcame"""
    cam = cv2.VideoCapture(0)
    suc, frame = cam.read()
    if suc:
        return frame
    else:
        raise ValueError("Can't access webcam!")

def load_test_img(test_filename='test/test_img.png', clear_test_img=False):
    """Load an image to test the face detector.

    :param test_filename: Relative path to the test image.
    :param clear_test_img: Boolean flag to clear the test image.

    :return: (M, N, 3) color opencv image array.
    """
    if os.path.exists(test_filename) and not clear_test_img:
        return cv2.imread(test_filename)
    elif os.path.exists(test_filename) and clear_test_img:
        logger.info('Clearing test file.')
        os.remove(test_filename)

    frame = capture_webcam_img()
    cv2.imwrite(test_filename, frame)
    return frame

def detect_face(img, classifier):
    """Detect a face in an image.

    :param img: (M, N, 3) color opencv image array.
    :param classifier: OpenCv CascadeClassifier for detecting faces.

    :return: (M, N, 3) color opencv image array with face highlighted.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray,
                                        scaleFactor=1.3,
                                        minNeighbors=5,
                                        minSize=(30, 30),
                                        flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    if len(faces) == 0:
        logger.warning('No faces detected!')
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return img


def main(argv=sys.argv[1:]):
    logger.info('entry point')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clear-test-img', action='store_true', default=False,
                        help='Clear the test image from disk and create a new one via webcam')

    args = parser.parse_args(argv)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    img = load_test_img(clear_test_img=args.clear_test_img)
    detect_img = detect_face(img, face_cascade)
    cv2.imshow('test', detect_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
