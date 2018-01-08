#!/bin/bash

# Copyright 2012-2015 Brno University of Technology (author: Karel Vesely), Daniel Povey
# Copyright 2017 Pranay Dighe (Idiap Research Institute)
# Apache 2.0

# Begin configuration section.
nnet=               # non-default location of DNN (optional)
feature_transform=  # non-default location of feature_transform (optional)
srcdir=             # non-default location of DNN-dir (decouples model dir from decode dir)
nj=40
cmd=run.pl

post_opts="--precision=2"

use_gpu="no" # yes|no|optionaly
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 6 ]; then
   echo "Usage: $0 [options] <energy> <data-dir> <nnet-dir> <alignment-file> <pca-dir> <out-dir>"
   echo "e.g.: $0 90 data/train exp/dnn1 exp/pca-mean-energy-dnn1 tri2b_train_alignment.ark data/pdf-posteriors/dnn1-train-energy90"
   echo ""
   echo "This script performs forward pass of the data through the neural network"
   echo " and performs posterior reconstruction using the supplied PCA transforms."
   echo " Energy to be conserved during reconstruction needs to be supplied."
   echo " Ground truth aligments are needed for supervised reconstuction. PCA transform"
   echo " are selected based on these alignments for framewise reconstruction."
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo ""
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --post-opts                                      # options for rounding off and storing PCA reconstructed posteriors"
   echo "  --nnet <nnet>                                    # non-default location of DNN (opt.)"
   echo "  --srcdir <dir>                                   # non-default dir with DNN/models, can be different"
   echo "                                                   # from parent dir of <decode-dir>' (opt.)"
   exit 1;
fi

energy=$1
data=$2
dir=$3
alignment=$4
pcadir=$5
outdir=$6

[ -z $srcdir ] && srcdir=$dir; 
sdata=$data/split$nj;

mkdir -p $outdir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $outdir/num_jobs

# Select default locations to model files (if not already set externally)
[ -z "$nnet" ] && nnet=$srcdir/final.nnet
[ -z "$feature_transform" -a -e $srcdir/final.feature_transform ] && feature_transform=$srcdir/final.feature_transform
#

# Check that files exist,
for f in $sdata/1/feats.scp $nnet $feature_transform $pcadir/pca.scp $pcadir/mean.scp $pcadir/energy.scp; do
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

# FORWARD PASS THE DATA AND PERFORM RECONSTRUCTION USING PCA TRANSFORMS
  #For senone 0
  $cmd JOB=1:$nj $outdir/log/fwd_pca.JOB.log \
    $KALDI_ROOT/src/pcabin/nnet-pca-forward --energy=$energy \
    --feature-transform=$feature_transform --use-gpu=$use_gpu "$nnet" "$feats" \
    ark:$alignment scp:$pcadir/pca.scp scp:$pcadir/mean.scp scp:$pcadir/energy.scp ark:- \| \
    $KALDI_ROOT/src/pcabin/posterior-to-post $post_opts ark:- \
    ark,scp:$outdir/post.JOB.ark,$outdir/post.JOB.scp

exit 0;
