
# UNet with CRF-as-RNN layer

This repository contains the implementation of a UNet with a CRF-as-RNN layer right before the final prediction. 
The code adapts the CRF-as-RNN implementation provided by M. Monteiro [here](https://github.com/MiguelMonteiro/CRFasRNNLayer), but we automatically extract the layer parameters, build the CRF-as-RNN layer, and integrate it in the UNet. This code is meant to make the original CRF-as-RNN implementation fully automatic, witout any need for user interaction. 

CRF-as-RNN is applied on the class scores predicted by the UNet on each image pixel: *i.e.*, right before applying the final softmax. After the layer, a softmax is used to extract each class probability.

--------------
## Notes:

The code was developed for semantic segmentation. We report code for running it on the [ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html). 

The UNet  segmentor with the additional CRF-as-RNN layer can be found under the folder `architectures`. Here, under the folder `architectures/layers` you can also find the CRF-as-RNN layer.

The experiment engine is inside `expriments/acdc/model.py`. This file contains the main class that is used to train on the ACDC dataset. 

--------------
## Requirements:

Code was developed and tested using TensorFlow 1.13 and 1.14. 
It was tested on NVIDIA GeForce GTX 1080 and TITAN Xp, with Driver Version: 440.95.01 and CUDA Version: 10.2.

