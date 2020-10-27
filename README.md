
# UNet with CRF-as-RNN layer

This repository contains the implementation of a UNet with a CRF-as-RNN layer right before the final prediction. 
The code adapts the CRF-as-RNN implementation provided by M. Monteiro [here](https://github.com/MiguelMonteiro/CRFasRNNLayer), but we automatically extract the layer parameters, build the CRF-as-RNN layer, and integrate it in the UNet. This code is meant to make the original CRF-as-RNN implementation fully automatic, witout any need for user interaction. 

CRF-as-RNN is applied on the class scores predicted by the UNet on each image pixel: *i.e.*, right before applying the final softmax. After the layer, a softmax is used to extract each class probability.

**************
## Before using the code:
1) Based on the TensorFlow version you are using, you may have to set the flag `-D_GLIBCXX_USE_CXX11_ABI` to 0 or 1. In 
our case we use `-D_GLIBCXX_USE_CXX11_ABI=1`; if you want to remove this behaviour you can comment the line:
   `target_compile_options(lattice_filter PUBLIC "-D_GLIBCXX_USE_CXX11_ABI=1")`
inside of the file: `architectures/layers/crf_as_rnn/permutohedral_lattice/CMakeLists.txt`. Refer to [this](https://github.com/google/sentencepiece/issues/293) github issue
for additional information.

2) Before using, remember to set the C++ and the CUDA compilers you want to use for building the layer inside the file: 
`architectures/layers/crf_as_rnn/permutohedral_lattice/build.sh`. 

Refer to `architectures/layers/crf_as_rnn/permutohedral_lattice/README.md` for additional details on the layer.

**************
## Common issues:
If your building fails, you may have to manually delete the file `architectures/layers/crf_as_rnn/permutohedral_lattice/config.txt`
before compiling the layer again.

**************
## Notes:

The code was developed for semantic segmentation. We report code for running it on the [ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html). 

The UNet  segmentor with the additional CRF-as-RNN layer can be found under the folder `architectures`. Here, under the folder `architectures/layers` you can also find the CRF-as-RNN layer.

The experiment engine is inside `expriments/acdc/model.py`. This file contains the main class that is used to train on the ACDC dataset. 

Refer to [this repository](https://github.com/gvalvano/multiscale-adversarial-attention-gates) for details in downloading/using the dataset.

**************
## Requirements:

Code was developed and tested using TensorFlow 1.13 and 1.14. 
It was tested on NVIDIA GeForce GTX 1080 and TITAN Xp, with Driver Version: 440.95.01 and CUDA Version: 10.2.
