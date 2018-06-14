from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 #para testar o que seria melhor

im = Image.open('Images/tif/085.tif')

# convert to numpy array
img = np.array(im)
Image.fromarray(img).save('orig.png')

# plt.imshow(img, cmap='gray')
# plt.show()

avg = cv2.blur(img, (5, 5))
Image.fromarray(avg).save('avg.png')

gauss = cv2.GaussianBlur(img, (5, 5), 0)
Image.fromarray(gauss).save('gauss.png')

median = cv2.medianBlur(img, 5)
Image.fromarray(median).save('median.png')

plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(avg, cmap='gray'), plt.title('Average')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(gauss, cmap='gray'), plt.title('Gaussian')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(median, cmap='gray'), plt.title('Median')
plt.xticks([]), plt.yticks([])
plt.show()
