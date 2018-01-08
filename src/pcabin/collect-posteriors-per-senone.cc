// nnetbin/compute-pca.cc

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

int main(int argc, char *argv[]) {
	using namespace kaldi;
	using namespace kaldi::nnet1;
	try {
	const char *usage =
		"Collect posteriors for a senone class by performing forward pass through a Neural Network.\n"
		"Needs ground truth alignment of senones on the input data.\n"
		"Parameter dataSize controls how many posteriors will be collected.\n"
		"To collect posteriors in log domain, either use --apply-log=true or use --no-softmax=true, but not both.\n"
		"\n"
		"Usage:	collect-posteriors-per-senone [options] <model-in> <feature-rspecifier> <alignment-rspecifier> <feature-wspecifier>\n"
		"e.g.: \n"
		" collect-posteriors-per-senone nnet ark:features.ark ark:alignment.ark ark:mlpoutput.ark\n";

	ParseOptions po(usage);

	std::string feature_transform;
	po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");

	int32 dataSize = 5000;
	po.Register("dataSize",&dataSize,"Maximum number of frames for computing principal components. Usually choose higher than dimension of posterior features.");

	int32 senone = 0;
	po.Register("senone",&senone,"Senone ID (indexed from 0) whose principal components have to be computed");

	bool correct_class = true;
	po.Register("correct-class",&correct_class,"Only pick correctly classified posterior frames as per MAP probability");

	bool no_softmax = false;
	po.Register("no-softmax", &no_softmax, "No softmax on MLP output (or remove it if found).");

	bool apply_log = false;
	po.Register("apply-log", &apply_log, "Transform MLP output to logscale");

	std::string use_gpu="no";
	po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

	using namespace kaldi;
	using namespace kaldi::nnet1;
	typedef kaldi::int32 int32;

	po.Read(argc, argv);

	if (po.NumArgs() != 4) {
		po.PrintUsage();
		exit(1);
	}

	std::string model_filename = po.GetArg(1),
				feature_rspecifier = po.GetArg(2),
				alignment_rspecifier = po.GetArg(3),
				feature_wspecifier = po.GetArg(4) ;		
 
	//Select the GPU
#if HAVE_CUDA==1
	KALDI_VLOG(2)<<"use_gpu="<<use_gpu;
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
	
	// disable dropout
	nnet_transf.SetDropoutRetention(1.0);
	nnet.SetDropoutRetention(1.0);

	kaldi::int64 tot_t = 0;

	SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
	RandomAccessInt32VectorReader alignment_reader(alignment_rspecifier);
	BaseFloatMatrixWriter feature_writer(feature_wspecifier);

	int32 dimension = nnet.OutputDim();

	CuMatrix<BaseFloat> feats, feats_transf, nnet_out;
	Matrix<BaseFloat> nnet_out_host;

	Matrix<BaseFloat> output;

	Timer time;
	double time_now = 0;
	int32 num_done = 0;
	Matrix<float> dataX;
	int32 currSize = 0;

	for (; !feature_reader.Done() && currSize < dataSize ; feature_reader.Next()) {
		// read
		if(alignment_reader.HasKey(feature_reader.Key())) {
			const std::vector<int32> &vec = alignment_reader.Value(feature_reader.Key());
			//Before performing forward pass on an utterance, we check the corresponding alignment for presence of
			//the required senone. Boolean flag is set to true if the senone is found in the alignments.
			bool flag = false;
			for (int32 r = 0; r < vec.size() ; r++) {
				if(senone == vec[r]) {
					flag = true;
					break;
				}
			}
			if(flag) {
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
				const std::vector<int32> &vec = alignment_reader.Value(feature_reader.Key());
				Vector<BaseFloat> frame_data; 
				for (int32 r = 0; r < vec.size() ; r++) {
					if(senone == vec[r]) {
						frame_data = nnet_out_host.Row(r);
						if(correct_class) {
							int32 max_element;
							frame_data.Max(&max_element);
							KALDI_VLOG(1) << "Max Element is : " << max_element << " and senone:	" << senone;
							if (senone == max_element) {
								KALDI_VLOG(1) << "Size of dataX is : " << dataX.NumRows() << " and " << dataX.NumCols();
								dataX.Resize(currSize + 1, dimension ,kCopyData);
								for (int32 c = 0; c < dimension ; c++) {
									dataX(currSize,c) = frame_data(c);
								}
								currSize = dataX.NumRows();
							}
						} else {
							KALDI_VLOG(1)<< "Adding one more frame.";
							dataX.Resize(currSize + 1, dimension ,kCopyData);
							for (int32 c = 0; c < dimension ; c++) {
								dataX(currSize,c) = frame_data(c);
							}
							currSize = dataX.NumRows();
						}							
					}
				}
			}				
		}

		// progress log,
		if (num_done % 100 == 0) {
		time_now = time.Elapsed();
		KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
						<< time_now/60 << " min; processed " << tot_t/time_now
						<< " frames per second.";
		}

		num_done++;
		tot_t += dataX.NumRows();
 	}
	
	// convert posteriors to log-posteriors
	if (apply_log) {
		dataX.ApplyLog();
	}
	 	
	
	//WriteDataCollected
	const std::string &senone_id = patch::to_string(senone);
	feature_writer.Write(senone_id, dataX);

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
