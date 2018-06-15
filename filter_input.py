from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
import time
import threading

def medianFilter(img, n):
	imgOut = np.zeros(img.shape, dtype=float)
	# indice central do filtro, equivalente ao padding necessario para aplica-lo na imagem
	center = int(n/2 - 0.5)
	# padding de bordas da imagem
	img = np.pad(img, ((center,), (center,)), 'edge')

	for i in range(center, img.shape[0] - center):
		iOut = i - center
		for j in range(center, img.shape[1] - center):
			jOut = j - center
			neighbourhood = img[i-center:i+center+1, j-center:j + center+1]  # janela do filtro (vizinhanca)
			median = np.median(neighbourhood)  # media dos pixels da vizinhanca
			# variancia local aproveita resultado anterior da media
			imgOut[iOut, jOut] = median

	imax = np.max(imgOut)
	imin = np.min(imgOut)
	minRange = 255 * imin / imax
	imgNorm = (imgOut - imin)/(imax - imin)
	imgNorm = minRange + (imgNorm*(255-minRange))
	imgNorm = imgNorm.astype(np.uint8)  # converte para unsigned int de 8 bits
	imgOut = imgNorm

	return imgOut

def gaussianFilter(img, n, sigma):
	imgOut = np.zeros(img.shape, dtype=np.complex)
	# kernel = np.zeros([n, n], dtype=float)
	# center = int(n/2 - 0.5)
	
	#  dx, dy = np.meshgrid(np.linspace(-2,2,5),np.linspace(-2,2,5), sparse=True) -> dx vetor, dy coluna
	# Calcula o kernel do filtro gaussiano
	def gaussian_kernel(n, sigma):
		kernel_2d = np.zeros([n, n], dtype=float)
		center = int(n/2 - 0.5)

		x2d, y2d = np.meshgrid(np.linspace(-center, center, n),
		                       np.linspace(-center, center, n), sparse=True)
		kernel_2d = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
		kernel_2d = kernel_2d / (2 * np.pi * sigma ** 2)

		norm_kernel_2d = kernel_2d/np.sum(kernel_2d)

		return norm_kernel_2d

	# Operacao de convolucao entre uma imagem f e filtro w
	def convolution_2d(f, w):
		central_index_x = int(w.shape[0]/2 - 0.5)
		central_index_y = int(w.shape[1]/2 - 0.5)
		g = np.zeros(f.shape, dtype=float)

		# Extensao da matriz com valores de borda para aplicar o filtro em todos os valores da imagem
		f = np.pad(f, ((central_index_x,), (central_index_y,)), 'edge')

		# Operacao de convolucao, percorrendo a imagem f com uma submatriz sub_f do
		# tamanho do filtro e utilizando estes valores para calcular os elementos
		# da imagem processda g.
		for x in range(g.shape[0]):
			for y in range(g.shape[1]):
				sub_f = f[x: x+w.shape[0], y: y+w.shape[1]]
				g[x, y] = np.sum(np.multiply(sub_f, w))

		return g

	kernel = gaussian_kernel(5,1) # ok!
	imgOut = convolution_2d(img, kernel)

	imax = np.max(imgOut)
	imin = np.min(imgOut)
	minRange = 255 * imin / imax
	imgNorm = (imgOut - imin)/(imax - imin)
	imgNorm = minRange + (imgNorm*(255-minRange))
	imgNorm = imgNorm.astype(np.uint8)  # converte para unsigned int de 8 bits
	imgOut = imgNorm
	
	return imgOut

# im = Image.open('Images/tif/085.tif')

# # convert to numpy array
# img = np.array(im)

# # apply some filter with window size 5
# filtered_img1 = gaussianFilter(img, 5, 0.025)
# filtered_imghalf = gaussianFilter(img, 5, 1)
# filtered_img3 = gaussianFilter(img, 5, 5)
# filtered_img25 = gaussianFilter(img, 5, 10)
# # Image.fromarray(filtered_img).convert('RGB').save('Images/Filtered/_gauss085.png')
# plt.subplot(221), plt.imshow(filtered_img25, cmap='gray')
# plt.subplot(222), plt.imshow(filtered_imghalf, cmap='gray')
# plt.subplot(223), plt.imshow(filtered_img1, cmap='gray')
# plt.subplot(224), plt.imshow(filtered_img3, cmap='gray')
# plt.show()

# pdb.set_trace()

files = [f for f in os.listdir('Images/tif')]
plt.ion()

start_time = time.time()
for f in files:
	im = Image.open('Images/tif/' + f)

	# convert to numpy array
	img = np.array(im)

	# apply some filter with window size 5
	filtered_img = medianFilter(img, 5)
	Image.fromarray(filtered_img).convert('RGB').save('Images/Filtered/Median/' + f[:3] + '.png')

print("--- %s sec for median on 30 images ---" % (time.time() - start_time))

start_time = time.time()
for f in files:
	im = Image.open('Images/tif/' + f)

	# convert to numpy array
	img = np.array(im)

	# apply some filter with window size 5
	filtered_img = gaussianFilter(img, 5, 1)
	Image.fromarray(filtered_img).convert('RGB').save('Images/Filtered/Gaussian/' + f[:3] + '.png')

print("--- %s sec for gaussian on 30 images ---" % (time.time() - start_time))






