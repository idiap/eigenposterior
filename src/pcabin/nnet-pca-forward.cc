// nnetbin/nnet-pca-forward.cc

// Copyright 2011-2013	Brno University of Technology (Author: Karel Vesely)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <limits>
#include <string> 
#include <cstdlib>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-pdf-prior.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"

#include <sstream>

namespace patch
{
	template < typename T > std::string to_string( const T& n )
	{
		std::ostringstream stm ;
		stm << n ;
		return stm.str() ;
	}
}

std::string replace(std::string &s,
		const std::string &toReplace,
		const std::string &replaceWith)
{
	return(s.replace(s.find(toReplace), toReplace.length(), replaceWith));
}


int main(int argc, char *argv[]) {
	using namespace kaldi;
	using namespace kaldi::nnet1;
	try {
	const char *usage =
		"Perform forward pass through Neural Network and apply the supplied PCA transform preserving the required amount of energy."
		"\n"
		"Usage:	nnet-pca-forward [options] <model-in> <feature-rspecifier> <alignment-rspecifier> <pca-transform-rspecifier> <mean-rspecifier> <energy-rspecifier> <feature-wspecifier>\n"
		"e.g.: \n"
		" nnet-pca-forward nnet ark:features.ark ark:alignments.ark scp:pca.scp scp:means.scp scp:energy.scp ark:mlpoutput.ark\n";

	ParseOptions po(usage);

	PdfPriorOptions prior_opts;
	prior_opts.Register(&po);

	std::string feature_transform;
	po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");

	int32 energy = 100;
	po.Register("energy",&energy,"Given amount of energy/covariance will be preserved while performing reconstruction using principal components");

	bool apply_log = true;
	po.Register("apply-log", &apply_log, "Transform MLP output to logscale. This is needed if the PCA transform were also learned in log domain");

	bool no_softmax = false;
	po.Register("no-softmax", &no_softmax, "No softmax on MLP output (or remove it if found). This is needed is the PCA transform were computed using posteriors from pre-softmax activation layer");

	bool apply_exp = true;
	po.Register("apply-exp", &apply_exp, "Transform the final PCA reconstructed features from logscale to posteriors");

	std::string use_gpu="no";
	po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

	po.Read(argc, argv);

	if (po.NumArgs() != 7) {
		po.PrintUsage();
		exit(1);
	}

	std::string model_filename = po.GetArg(1),
				feature_rspecifier = po.GetArg(2),
				alignment_rspecifier = po.GetArg(3),
				transform_rspecifier = po.GetArg(4),
				mean_rspecifier = po.GetArg(5),
				energy_rspecifier = po.GetArg(6),
				feature_wspecifier = po.GetArg(7);
		
	using namespace kaldi;
	using namespace kaldi::nnet1;
	typedef kaldi::int32 int32;
		
	float eps= 2.2204e-16;
	//Select the GPU
#if HAVE_CUDA==1
	CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

	Nnet nnet_transf;
	if (feature_transform != "") {
		nnet_transf.Read(feature_transform);
	}

	Nnet nnet;
	nnet.Read(model_filename);
	//optionally remove softmax
	if (no_softmax && nnet.GetComponent(nnet.NumComponents()-1).GetType() ==
		kaldi::nnet1::Component::kSoftmax) {
		KALDI_LOG << "Removing softmax from the nnet " << model_filename;
		nnet.RemoveComponent(nnet.NumComponents()-1);
	}
	//check for some non-sense option combinations
	if (apply_log && no_softmax) {
		KALDI_ERR << "Nonsense option combination : --apply-log=true and --no-softmax=true";
	}
	if (apply_log && nnet.GetComponent(nnet.NumComponents()-1).GetType() !=
		kaldi::nnet1::Component::kSoftmax && nnet.GetComponent(nnet.NumComponents()-1).GetType() != kaldi::nnet1::Component::kSoftmaxT) {
		KALDI_ERR << "Used --apply-log=true, but nnet " << model_filename 
				<< " does not have <softmax> as last component!";
	}
	
	PdfPrior pdf_prior(prior_opts);
	if (prior_opts.class_frame_counts != "" && (!no_softmax && !apply_log)) {
		KALDI_ERR << "Option --class-frame-counts has to be used together with "
				<< "--no-softmax or --apply-log";
	}

	// disable dropout
	nnet_transf.SetDropoutRetention(1.0);
	nnet.SetDropoutRetention(1.0);

	kaldi::int64 tot_t = 0;

	SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
	RandomAccessInt32VectorReader alignment_reader(alignment_rspecifier);
	RandomAccessBaseFloatMatrixReader transform_reader(transform_rspecifier);
	RandomAccessBaseFloatVectorReader mean_reader(mean_rspecifier);
	SequentialInt32VectorReader energy_reader(energy_rspecifier);
	BaseFloatMatrixWriter feature_writer(feature_wspecifier);



	int32 dimension = nnet.OutputDim();


	Vector<float> energies(dimension);
	for(;!energy_reader.Done(); energy_reader.Next()){
		const std::vector<int32> &v = energy_reader.Value();
		int32 senone = std::atoi(energy_reader.Key().c_str());
		energies(senone)=v[energy];
	}

	Timer time;
	double time_now = 0;
	int32 num_done = 0;
	// iterate over all feature files
	//	for (; !feature_reader.Done() || !alignment_reader.Done(); feature_reader.Next()) {
	for (; !feature_reader.Done(); feature_reader.Next()) {
		// read
		Matrix<BaseFloat> nnet_out_host;
		CuMatrix<BaseFloat> feats, feats_transf, nnet_out;
		Matrix<BaseFloat> output;
		const Matrix<BaseFloat> &mat = feature_reader.Value();
			
		KALDI_VLOG(2) << "Processing utterance " << num_done+1 
					<< ", " << feature_reader.Key() 
					<< ", " << mat.NumRows() << "frm";

		//check for NaN/inf
		BaseFloat sum = mat.Sum();
		if (!KALDI_ISFINITE(sum)) {
			KALDI_ERR << "NaN or inf found in features of " << feature_reader.Key();
		}
		
		// push it to gpu
		feats = mat;
		// fwd-pass
		nnet_transf.Feedforward(feats, &feats_transf);
		nnet.Feedforward(feats_transf, &nnet_out);
		
		// convert posteriors to log-posteriors
		if (apply_log) {
			nnet_out.ApplyLog();
		}
	 
		// subtract log-priors from log-posteriors to get quasi-likelihoods
		if (prior_opts.class_frame_counts != "" && (no_softmax || apply_log)) {
			pdf_prior.SubtractOnLogpost(&nnet_out);
		}
	 
		//download from GPU
		nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols());
		output.Resize(nnet_out.NumRows(), nnet_out.NumCols());
		nnet_out.CopyToMat(&nnet_out_host);

		//check for NaN/inf
		for (int32 r = 0; r < nnet_out_host.NumRows(); r++) {
			for (int32 c = 0; c < nnet_out_host.NumCols(); c++) {
				BaseFloat val = nnet_out_host(r,c);
				if (val != val) KALDI_ERR << "NaN in NNet output of : " << feature_reader.Key();
				if (val == std::numeric_limits<BaseFloat>::infinity())
					KALDI_ERR << "inf in NNet coutput of : " << feature_reader.Key();
			}
		}

		output = nnet_out_host;
		if(alignment_reader.HasKey(feature_reader.Key())) {
			const std::vector<int32> &vec = alignment_reader.Value(feature_reader.Key());
			for (int32 r = 0; r < nnet_out_host.NumRows(); r++) {
				Vector<BaseFloat> frame_data; 
				frame_data.Resize(nnet_out_host.NumCols());
				for (int32 c = 0; c < nnet_out_host.NumCols(); c++) {
					frame_data(c)=nnet_out_host(r,c);
				}	
				std::string senone = patch::to_string(vec[r]);
		 		Vector<BaseFloat> temp_data;
		 		Vector<BaseFloat> pure_frame_data;
				
				if(transform_reader.HasKey(senone) && mean_reader.HasKey(senone)) {
					temp_data.Resize(energies(std::atoi(senone.c_str())));
					KALDI_VLOG(2) << "PCA for senone " << senone << " found.";
					SubMatrix<BaseFloat> trans_pca(transform_reader.Value(senone),0, dimension,0,(int32) energies(std::atoi(senone.c_str())));
					KALDI_VLOG(2) << energies(std::atoi(senone.c_str())) << " principal components kept for senone " << senone <<".";
					frame_data.AddVec(-1.0,mean_reader.Value(senone));
					temp_data.AddMatVec(1.0,trans_pca,kTrans,frame_data,0.0);
					pure_frame_data = mean_reader.Value(senone);
					pure_frame_data.AddMatVec(1.0,trans_pca,kNoTrans,temp_data,1.0);
					for (int32 c = 0; c < nnet_out_host.NumCols(); c++) {
						output(r,c)=pure_frame_data(c);
					}
				}
				else {
					KALDI_VLOG(2) << "PCA for senone " << senone << " NOT found.";
					for (int32 c = 0; c < nnet_out_host.NumCols(); c++) {
						output(r,c)=log(eps);
					}
					output(r,std::atoi(senone.c_str())) = 0.0;
				}
			}
		}
		else{
			KALDI_VLOG(2) << "Alignment not found";
		}
			
		// write
		if(apply_exp){
			output.ApplyExp();	
		}

		feature_writer.Write(feature_reader.Key(), output);

		// progress log
		if (num_done % 100 == 0) {
		time_now = time.Elapsed();
		KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
						<< time_now/60 << " min; processed " << tot_t/time_now
						<< " frames per second.";
		}
		num_done++;
		tot_t += mat.NumRows();
	}
	
	// final message
	KALDI_LOG << "Done " << num_done << " files" 
				<< " in " << time.Elapsed()/60 << "min," 
				<< " (fps " << tot_t/time.Elapsed() << ")"; 

#if HAVE_CUDA==1
	if (kaldi::g_kaldi_verbose_level >= 1) {
		CuDevice::Instantiate().PrintProfile();
	}
#endif

	if (num_done == 0) return -1;
	return 0;
	} catch(const std::exception &e) {
	KALDI_ERR << e.what();
	return -1;
	}
}
