from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import time

def medianFilter(img, n):
	imgOut = np.zeros(img.shape, dtype=float)
	# indice central do filtro, equivalente ao padding necessario para aplica-lo na imagem
	center = int(n/2 - 0.5)
	# padding circular da imagem
	img = np.pad(img, ((center,), (center,)), 'wrap')

	for i in range(center, img.shape[0] - center):
		iOut = i - center
		for j in range(center, img.shape[1] - center):
			jOut = j - center
			neighbourhood = img[i-center:i+center+1, j-center:j +
                            center+1]  # janela do filtro (vizinhanca)
			median = np.median(neighbourhood)  # media dos pixels da vizinhanca
			# variancia local aproveita resultado anterior da media
			imgOut[iOut, jOut] = median

	return imgOut

# open 5 images from tif files
init = 85
total = 10

start_time = time.time()
for i in range(10):
	im = Image.open('Images/tif/0'+ str(init+i) + '.tif')

	# convert to numpy array
	img = np.array(im)

	# apply some filter with window size 5
	filtered_img = medianFilter(img, 5)
	Image.fromarray(filtered_img).convert('RGB').save('Images/Filtered/0'+ str(init+i) + '.png')

print("--- %s sec for median on 10 images ---" % (time.time() - start_time))


