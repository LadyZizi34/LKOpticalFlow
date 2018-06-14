from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb

def localNoiseRedFilter(img, sigma, n):
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
			avgPixels = np.mean(neighbourhood)  # media dos pixels da vizinhanca
			# variancia local aproveita resultado anterior da media
			localSigma = np.mean((neighbourhood - avgPixels) ** 2)
			if localSigma == 0:  # resultaria em uma divisao por 0, portanto nao altera o pixel original
				imgOut[iOut, jOut] = img[i, j]
				continue
			else:  # calculo do valor resultante do pixel
				imgOut[iOut, jOut] = img[i, j] - \
				    (sigma/localSigma) * (img[i, j] - avgPixels)

	return imgOut

# open image from tif file
im = Image.open('Images/tif/085.tif')

# convert to numpy array
img = np.array(im)

# apply some filter with window size 5
filtered_img = localNoiseRedFilter(img, 5, 0.025)
Image.fromarray(filtered_img).convert('RGB').save('Images/Filtered/085.png')



