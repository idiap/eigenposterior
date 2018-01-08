#!/bin/bash/
# Copyright  2017 Pranay Dighe (Idiap Research Institute)
# Apache 2.0.
#

. cmd.sh
. path.sh
set -e

#Training command
cmd="utils/queue.pl -V"
nnet_cmd="utils/queue.pl -V -l q_gpu"

#Principal Components learning options
datasize=5000
num_senones=4000

#PCA reconstruction options
nj=40
pca_energy=90
post_opts="--precision=2"

#Folder and File Locations
traindir=data/train
cvdir=data/cv
testdir=data/test
graphdir=exp/tri2/graph

nnetdir=exp/tri3_dn
pcadir=exp/tri3_dn_pca_mean_energy
alidir=exp/tri2_train_ali
alignment=${alidir}/alignment.ark
trainpostdir=data/posteriors/tri3_dn_train_pca${pca_energy}
cvpostdir=data/posteriors/tri3_dn_cv_pca${pca_energy}

#Create alignment
ali-to-pdf $alidir/final.mdl "ark:gunzip -c $alidir/ali.*.gz |" ark,t:$alignment

#Estimate principal components for each senone class
pca/estimate_pca.sh --cmd "$cmd" --datasize $datasize $num_senones $traindir $nnetdir $alignment $pcadir

#Create scps for PC transforms, senone class means and energy percentile vectors to be used for nnet-pca-forward in next step
cat $pcadir/pca_*.scp > $pcadir/pca.scp
cat $pcadir/mean_*.scp > $pcadir/mean.scp
cat $pcadir/energy_*.scp > $pcadir/energy.scp

#Perform forward pass of data and reconstruct DNN posteriors using PCA transforms
pca/pca_forward.sh --cmd "$cmd" --nj $nj --post-opts "$post_opts" $pca_energy $traindir $nnetdir $alignment $pcadir $trainpostdir
pca/pca_forward.sh --cmd "$cmd" --nj $nj --post-opts "$post_opts" $pca_energy $cvdir $nnetdir $alignment $pcadir $cvpostdir

#Create scp files for train data and cv data posteriors (or soft targets)
cat $trainpostdir/*.scp > ${alidir}/tri3_dn_pca${pca_energy}.scp
analyze-post-counts --counts-dim=$num_senones scp:${alidir}/tri3_dn_pca${pca_energy}.scp - > $trainpostdir/ali_train_pdf.counts

cat $cvpostdir/*.scp > ${alidir}/tri3_dn_pca${pca_energy}.scp #Note: append cv soft targets to the same scp file

#Train a new DNN acoustic model using soft targets created above
$nnet_cmd exp/tri3_dn_pca95/_train_nnet.log steps/train_nnet.sh --hid-dim 1200 --norm-vars true --apply-cmvn true --splice 4 --feat-type plain --delta_order 2 --learn-rate 0.008 --labels scp:${alidir}/tri3_dn_pca${pca_energy}.scp data/train data/cv data/lang dummy dummy exp/tri3_dn_pca95

cp $trainpostdir/ali_train_pdf.counts exp/tri3_dn_pca95/ali_train_pdf.counts
cp exp/tri3_dn/final.mdl exp/tri3_dn_pca95/.

#Decode test data
steps/decode_nnet.sh --use_gpu "no" --parallel_opts "" --nj $nj  --cmd $cmd exp/tri3_dn/graph data/test exp/tri3_dn_pca${pca_energy}
~            
