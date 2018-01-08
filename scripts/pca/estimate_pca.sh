#!/bin/bash

# Copyright 2012-2015 Brno University of Technology (author: Karel Vesely), Daniel Povey
# Copyright 2017 Pranay Dighe (Idiap Research Institute)
# Apache 2.0

# Begin configuration section.
nnet=               # non-default location of DNN (optional)
feature_transform=  # non-default location of feature_transform (optional)
srcdir=             # non-default location of DNN-dir (decouples model dir from decode dir)

cmd=run.pl

collect_posterior_opts="--no-softmax=false --apply-log=true"
datasize=5000 #Number of posterior frames to be used for estimating PCA
correct_class=true

use_gpu="no" # yes|no|optionaly
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 5 ]; then
   echo "Usage: $0 [options] <num_senones> <data-dir> <nnet-dir> <alignment> <pca-output-dir>"
   echo "e.g.: $0 4000 data/train exp/dnn1 data/pdf-alignments/tri2b_train_alignments.ark exp/pca-mean-energy-dnn1"
   echo ""
   echo "This script collects posteriors senonewise and learns PCA transforms"
   echo " and related statistics for each senone. Posterior collection and Principal Component"
   echo " learning is done as parallel jobs, one job for each senone."
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo ""
   echo "  --nnet <nnet>                                    # non-default location of DNN (opt.)"
   echo "  --srcdir <dir>                                   # non-default dir with DNN/models, can be different"
   echo "  --datasize                                      # number of posteriors to be collected for estimating principal components"
   echo "                                                   # from parent dir of <decode-dir>' (opt.)"
   exit 1;
fi

num_senones=$1
data=$2
dir=$3
alignment=$4
outdir=$5

[ -z $srcdir ] && srcdir=$dir; 

mkdir -p $outdir/log

echo $num_senones > $outdir/num_jobs

# Select default locations to model files (if not already set externally)
[ -z "$nnet" ] && nnet=$srcdir/final.nnet
[ -z "$feature_transform" -a -e $srcdir/final.feature_transform ] && feature_transform=$srcdir/final.feature_transform
#

# Check that files exist,
for f in $data/feats.scp $nnet $feature_transform $alignment; do
  [ ! -f $f ] && echo "$0: missing file $f" && exit 1;
done

# PREPARE FEATURE EXTRACTION PIPELINE
# import config,
cmvn_opts=
delta_opts=
D=$srcdir
[ -e $D/norm_vars ] && cmvn_opts="--norm-means=true --norm-vars=$(cat $D/norm_vars)" # Bwd-compatibility,
[ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
[ -e $D/delta_order ] && delta_opts="--delta-order=$(cat $D/delta_order)" # Bwd-compatibility,
[ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
#
# Create the feature stream,
feats="ark,s,cs:copy-feats scp:$data/feats.scp ark:- |"
# apply-cmvn (optional),
[ ! -z "$cmvn_opts" -a ! -f $data/cmvn.scp ] && echo "$0: Missing $data/cmvn.scp" && exit 1
[ ! -z "$cmvn_opts" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp ark:- ark:- |"
# add-deltas (optional),
[ ! -z "$delta_opts" ] && feats="$feats add-deltas $delta_opts ark:- ark:- |"

# COLLECT POSTERIORS AND ESTIMATE PCA
  #For senone 0
  $cmd JOB=1 $outdir/log/pca_senone0.JOB.log \
    $KALDI_ROOT/src/pcabin/collect-posteriors-per-senone --senone=0 $collect_posterior_opts --dataSize=$datasize --correct-class=$correct_class \
    --feature-transform=$feature_transform --use-gpu=$use_gpu "$nnet" "$feats" ark:$alignment ark:- \| \
    $KALDI_ROOT/src/pcabin/est-pca-mean-energy --binary=false --senone-id=0 ark:- \
    ark,scp:$outdir/pca_0.ark,$outdir/pca_0.scp ark,scp:$outdir/mean_0.ark,$outdir/mean_0.scp ark,scp:$outdir/energy_0.ark,$outdir/energy_0.scp

  #For rest of the senones
  $cmd JOB=1:$(expr $num_senones - 1) $outdir/log/pca.JOB.log \
    $KALDI_ROOT/src/pcabin/collect-posteriors-per-senone --senone=JOB $collect_posterior_opts --dataSize=$datasize --correct-class=$correct_class \
    --feature-transform=$feature_transform --use-gpu=$use_gpu "$nnet" "$feats" ark:$alignment ark:- \| \
    $KALDI_ROOT/src/pcabin/est-pca-mean-energy --binary=false --senone-id=JOB ark:- \
    ark,scp:$outdir/pca_JOB.ark,$outdir/pca_JOB.scp ark,scp:$outdir/mean_JOB.ark,$outdir/mean_JOB.scp ark,scp:$outdir/energy_JOB.ark,$outdir/energy_JOB.scp

exit 0;
