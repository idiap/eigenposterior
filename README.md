Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
Written by Pranay Dighe <pranay.dighe@idiap.ch>,

# 01/12/2017

## General Information
This is the README to the Eigenposterior (Senone Class Principal Components) based approach
for purifying DNN posterior estimates . Purified posteriors can be used as soft target training
of better DNN acoustic models. The system is based in Kaldi tookit for speech recognition.
This README contains information about the package, implementation
details, installation and compilation.

## Compilation

To compile the package simply follow the 2 steps

1. export the path to kaldi source in the environment variable $KALDI_ROOT

```
export KALDI_ROOT=/home/username/kaldi-trunk/
```

2. Run make in the src/ directory

```
cd src/
make
```

Now, the binaries should have been created in the src/pcabin/
folder.

## Kaldi recipes

There are two recipes in the scripts/pca folder:

1) estimate_pca.sh: This recipe collects posteriors senonewise and learns PCA transforms
 and related statistics for each senone. Posterior collection and Principal Component
 learning is done as parallel jobs, one job for each senone.

2) pca_forward.sh: This recipe performs forward pass of the data through the neural network
 and performs posterior reconstruction using the supplied PCA transforms.
 Energy to be conserved during reconstruction needs to be supplied.
 Ground truth aligments are needed for supervised reconstuction. PCA transform
 are selected based on these alignments for framewise reconstruction.

3) run.sh: An example recipe to train a DNN acoustic model from  PCA-purified soft targets.

## References

This setup is based on:

[1] P. Dighe, A. Asaei, H. Bourlard "Exploiting Eigenposteriors for Semi-supervised Training of DNN AcousticModels with Sequence Discrimination",  in Interspeech 2017, Stockholm, Sweden.

[2] P. Dighe, A. Asaei, H. Bourlard "Low-rank and Sparse Soft Targets to Learn Better DNN Acoustic Models",  in ICASSP 2017, New Orleans, USA.

[3] P. Dighe, G. Luyet, A. Asaei and H. Bourlard “Exploiting Low-dimensional Structures to enhance DNN Based Acoustic Modeling in Speech Recognition”, in ICASSP 2016, Shanghai, China.

Resources on PCA:

[4] J. Shlens "A tutorial on principal component analysis." arXiv preprint arXiv:1404.1100 (2014).

KALDI Speech Recognition Toolkit:

[5]  D. Povey, A. Ghoshal, G. Boulianne, L. Burget, O. Glembek, N. Goel, M. Hannemann, P. Motl´ıcek, Y. Qian, P. Schwarz ˇ et al., “The kaldi speech recognition toolkit,” 2011.
