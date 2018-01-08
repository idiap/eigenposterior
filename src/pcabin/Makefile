
all:

EXTRA_CXXFLAGS = -Wno-sign-compare  -I${KALDI_ROOT}/src
include ${KALDI_ROOT}/src/kaldi.mk
LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = posterior-to-post est-pca-mean-energy analyze-post-counts collect-posteriors-per-senone nnet-pca-forward

OBJFILES =

TESTFILES =

ADDLIBS = ${KALDI_ROOT}/src/nnet/kaldi-nnet.a ${KALDI_ROOT}/src/cudamatrix/kaldi-cudamatrix.a \
					${KALDI_ROOT}/src/decoder/kaldi-decoder.a ${KALDI_ROOT}/src/lat/kaldi-lat.a ${KALDI_ROOT}/src/lm/kaldi-lm.a \
          ${KALDI_ROOT}/src/fstext/kaldi-fstext.a ${KALDI_ROOT}/src/hmm/kaldi-hmm.a \
          ${KALDI_ROOT}/src/transform/kaldi-transform.a ${KALDI_ROOT}/src/gmm/kaldi-gmm.a \
          ${KALDI_ROOT}/src/tree/kaldi-tree.a ${KALDI_ROOT}/src/util/kaldi-util.a ${KALDI_ROOT}/src/thread/kaldi-thread.a \
          ${KALDI_ROOT}/src/matrix/kaldi-matrix.a ${KALDI_ROOT}/src/base/kaldi-base.a



include ${KALDI_ROOT}/src/makefiles/default_rules.mk
