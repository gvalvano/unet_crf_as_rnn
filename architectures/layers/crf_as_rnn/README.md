# CRF-as-RNN layer

Implementation adapted from:
https://github.com/MiguelMonteiro/CRFasRNNLayer

Implements Conditional Random Fields as Recurrent Neural Networks as in the repository from the original authors, that
can be found at: https://github.com/sadeepj/crfasrnn_keras .

************
# HOW TO USE:

1) update parameters for compiling the layer as you need: 'nano permutohedral_lattice/build.sh'

2) parameters to update are:
   SPATIAL_DIMS=2           ---> it means that it is a 2D image:
                                e.g. placeholder with shape [None, W, H, n_channels] has 2
                                e.g. placeholder with shape [None, W, H, D, n_channels] has 3
   INPUT_CHANNELS=4         ---> number of channels that the input to the layer has (n_channels)
   REFERENCE_CHANNELS=1     ---> number of channels of the reference (e.g. the associated input image)

3) compile: 'sh build.sh'

4) ...and you're ready to go!

************
# NOTA BENE:

1) Based on the TensorFlow version you are using, you may have to set the flag `-D_GLIBCXX_USE_CXX11_ABI` to 0 or 1. In 
our case we use `-D_GLIBCXX_USE_CXX11_ABI=1`; if you want to remove this behaviour you can comment the line:
   `target_compile_options(lattice_filter PUBLIC "-D_GLIBCXX_USE_CXX11_ABI=1")`
inside of the file: `architectures/layers/crf_as_rnn/permutohedral_lattice/CMakeLists.txt`. Refer to [this](https://github.com/google/sentencepiece/issues/293) github issue
for additional information.

2) Before using, remember to set the C++ and the CUDA compilers inside the file: 
`architectures/layers/crf_as_rnn/permutohedral_lattice/build.sh`

************
## References:

@article{monteiro2018conditional,
  title={Conditional random fields as recurrent neural networks for 3d medical imaging segmentation},
  author={Monteiro, Miguel and Figueiredo, M{\'a}rio AT and Oliveira, Arlindo L},
  journal={arXiv preprint arXiv:1807.07464},
  year={2018}
}

@inproceedings{crfasrnn_ICCV2015,
    author = {Shuai Zheng and Sadeep Jayasumana and Bernardino Romera-Paredes and Vibhav Vineet and
    Zhizhong Su and Dalong Du and Chang Huang and Philip H. S. Torr},
    title  = {Conditional Random Fields as Recurrent Neural Networks},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year   = {2015}
}
