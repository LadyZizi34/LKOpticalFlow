# Identificação de movimento em imagens através de fluxo óptico

Autora: Laíse Aquino - número USP 7986924

Descrição:

O projeto consiste na implementação do algoritmo de fluxo óptico Lucas-Kanade para identificação de movimento em uma sequência de imagens. O objetivo é reconhecer a direção em que elementos da imagem estão se movendo, apresentando essa informação na forma de cores atribuídas aos pixels de um frame que representam o sentido do movimento de cada um.

As imagens utilizadas para teste foram retiradas do conjunto "UCSD Anomaly Detection Dataset - Peds2", que mostram uma sequência de fotos tiradas de uma rua com sentido paralelo à câmera.

//exemplo de sequencia
//http://svcl.ucsd.edu/projects/anomaly/dataset.htm

Método:

Para cada frame da sequência, primeiramente realiza-se a suavização da imagem por um filtro de redução de ruído aplicado apenas ao canal V do sistema de cores HSV, visando evitar a propagação de possíveis ruídos sem introduzir grandes distorções na imagem. Em seguida, caso as imagens utilizadas sejam coloridas, estas são convertidas para escala de cinza antes da aplicação do algoritmo de Lucas-Kanade, que resulta em uma série de vetores de movimento. Por fim, são atribuídas cores a estes vetores de acordo com sua direção, seguindo as posições padronizadas pela roda de cores RGB. 
