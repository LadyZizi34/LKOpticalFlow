# Identificação de Movimento em Imagens Através de Fluxo Ótico
Laíse Aquino - número USP 7986924

## Descrição

O projeto consiste na implementação do algoritmo de fluxo ótico Lucas-Kanade para identificação de movimento em uma sequência de imagens. O objetivo é reconhecer a direção em que elementos da imagem estão se movendo, apresentando essa informação na forma de cores atribuídas aos pixels de um frame que representam o sentido do movimento de cada um.

As imagens utilizadas para teste foram retiradas do conjunto "UCSD Anomaly Detection Dataset - Peds2", que mostram uma sequência de fotos tiradas de uma rua com sentido paralelo à câmera.

![085](https://user-images.githubusercontent.com/6940966/41264053-eca43e90-6dc0-11e8-8a7c-8282b1b0f811.png)
![105](https://user-images.githubusercontent.com/6940966/41264054-ecc3a41a-6dc0-11e8-8291-e3bfa735088c.png)

## Método

Para cada frame da sequência, primeiramente realiza-se a suavização da imagem visando evitar a propagação de possíveis ruídos sem prejudicar a identificação dos elementos. Em seguida, é aplicado o algoritmo de Lucas-Kanade, que resulta em uma série de vetores de movimento. Por fim, são atribuídas cores a estes vetores de acordo com sua direção, seguindo as posições padronizadas pela roda de cores RGB. 

## Execução

Durante o projeto foram desenvolvidos os módulos *filter_input*, *lucas-kanade* e *farneback*, sendo o último apenas para comparação dos resultados com um método de fluxo ótico já implementado na biblioteca OpenCV. Demais módulos foram adicionados como dependências ao método de captura das imagens.
O módulo *filter_input* normaliza, aplica filtros, converte e salva as imagens originais em um formato pronto para ser aplicado ao algoritmo de fluxo ótico. Todas as imagens originais e filtradas já estão carregadas no repositório, logo não é necessário executar este script.
A execução do módulo *farneback* mostra o resultado desejado, exibindo de forma contínua o fluxo calculado sobre as imagens de teste. Já o módulo *lucas-kanade* utiliza as mesmas ferramentas de carregamento e visualização do módulo anterior, inerentes à biblioteca OpenCV, mas substitui o método pela implementação do algoritmo de Lucas-Kanade. Os resultados obtidos são melhor visualizados após um período de execução, quando são processadas imagens pertencentes ao meio da sequência.

## Referências

Imagens: http://svcl.ucsd.edu/projects/anomaly/dataset.htm

Algoritmo: https://www.datasciencecentral.com/profiles/blogs/implementing-lucas-kanade-optical-flow-algorithm-in-python

Exemplo OpenCV: https://docs.opencv.org/3.2.0/d7/d8b/tutorial_py_lucas_kanade.html
