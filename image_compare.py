
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2


# Read while cadillac xt4 image
original = cv2.imread("images/2020_cadillac_xt4_angularfront.jpg")
#second = cv2.imread("images/2020_cadillac_xt4_angularfront.jpg")
second = cv2.imread("images/cadillac_black.jpg")

# Convert it to grayscale
grayA = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

if(score == 1.0):
    print('Both images are exactly same')
else:
    print("Images are different")