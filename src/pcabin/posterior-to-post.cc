// bin/fwdpass-post-to-gpost.cc

// Copyright 2011-2012 Johns Hopkins University (Author: Daniel Povey)  Chao Weng
// Copyright 2017 Pranay Dighe (Idiap Research Institute)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/posterior.h"
#include <iostream>
#include <fstream>
#include <queue>
#include <vector>
#include <utility>

using namespace kaldi;
using namespace std;


//To compare a pair <double, int> by values stored in pair.first (the double value)
struct CompareByFirst {
	bool operator()(pair<double, int> const & a, pair<double, int> const & b)
	{ return a.first < b.first; }
};


int main(int argc, char *argv[]) {
  try {
	typedef kaldi::int32 int32;  

	const char *usage =
		"Convert DNN outputs from kaldi feats format to post format. For rounding off the probabilities,\n"
		"atleast one of precision, percentile and topN parameter should be non-zero. Default is precision=2.\n"
		"Set roundOff to false for converting DNN outputs as they are in post format (Warning: large amount of space required).\n"
		"\n"
		"Usage: posterior-to-post <posterior-rspecifier> <post-wspecifier>\n";

	ParseOptions po(usage);

	int32 precision = 2;
	po.Register("precision", &precision, "For keeping precision upto N places after decimal.");

	int32 percentile = 0;
	po.Register("percentile", &percentile, "For keeping N percentile probability in each posterior frame."); 

	int32 topN = 0;
	po.Register("topN", &topN, "For keeping topN probabilities in each posterior frame."); 

	bool roundOff = true;
	po.Register("roundOff", &roundOff, "Make it false for no rounding off."); 

	bool apply_exp = false;
	po.Register("apply-exp", &apply_exp, "Transform to exponent scale in case the input posterior features are in log domain.");

	po.Read(argc, argv);

	if (po.NumArgs() != 2) {
	  po.PrintUsage();
	  exit(1);
	}

	if (!precision && !percentile && !topN && roundOff) {
	  po.PrintUsage();
	  exit(1);
	}
	  
	std::string post_rspecifier = po.GetArg(1),
		post_wspecifier = po.GetArg(2);

	SequentialBaseFloatMatrixReader  posterior_reader(post_rspecifier);
	PosteriorWriter posterior_writer(post_wspecifier); 

	int32 num_done = 0;
	float eps = 2.2204e-16;
	for (; !posterior_reader.Done(); posterior_reader.Next()) {
		std::string key = posterior_reader.Key();
		Matrix<BaseFloat> feats = posterior_reader.Value();

		Posterior utt_post;
		int32 numrows = feats.NumRows(),
		numcols = feats.NumCols();
		utt_post.resize(numrows);

		Matrix<BaseFloat> mat(feats.NumRows(), feats.NumCols());
		mat.CopyFromMat(feats, kNoTrans);

		if(apply_exp){
			mat.ApplyExp();
		}

		//For rounding off probabilities to save storage space
		if (roundOff) {
			//For keeping precision upto 2 decimal places
			if (precision) {
				float digits = pow(10,precision);
				for(int32 r=0; r<numrows; r++){
					for(int32 c=0; c<numcols; c++){
						mat(r,c) = roundf(mat(r,c) * digits) / digits;
				  }
				}
			}
			//For keeping 98 percentile probability
			else if (percentile) {	
				float probsum = 0.0;
				float maxprobsum = percentile/100.0;
				for(int32 r=0; r<numrows; r++){
					SubVector<BaseFloat> frame = mat.Row(r);
					priority_queue<pair<double, int>, vector< pair<double,int> >, CompareByFirst > q;
					for (int32 c = 0; c < frame.Dim(); ++c) {
						q.push(std::pair<double, int>(frame(c), c));
						mat(r,c) = 0.0;
					}
					probsum = 0.0; // number of indices we need									
					for (int32 c = 0; probsum < maxprobsum && c < numcols; c++) {
						mat(r,q.top().second) = q.top().first;
						probsum += q.top().first;
						q.pop();
					}
				}
			}
			//For top N probabilities
			else if (topN) {
				for(int32 r=0; r<numrows; r++){
		  		SubVector<BaseFloat> frame = mat.Row(r);
					priority_queue<pair<double, int>, vector< pair<double, int> >, CompareByFirst > q;
					for (int32 c = 0; c <numcols; ++c) {
						q.push(std::pair<double, int>(frame(c),c));
					}
					for (int32 c = 0; c < topN; ++c) {
			  	  mat(r,q.top().second) = q.top().first;
				  	q.pop();
					}
				}
			}

			//Normalize probabilities to sum upto 1 framewise	
  		Vector<BaseFloat> ones(mat.NumCols());
			ones.Set(1.0);
			Vector<BaseFloat> sum(mat.NumRows());
			sum.AddMatVec(1.0, mat, kNoTrans, ones, 0.0);
			sum.Add(eps);
 		  sum.ApplyPow(-1);
	 		mat.MulRowsVec(sum);
		}
						
	  for(int32 r = 0; r < numrows; r++) {
		  for(int32 c = 0; c < numcols; c++) {
				BaseFloat val = mat(r,c);
		   	if (val <0.0 || val > 1.0) KALDI_ERR << "Some other than probabilities in : " << key;
			 	if (val != val) KALDI_ERR << "NaN in NNet output of : " << key << "and Value is at row " << r << " and column " << c;
				if (val == std::numeric_limits<BaseFloat>::infinity()) KALDI_ERR << "inf in NNet coutput of : " << key;
	  	  if (val > 0.0) utt_post[r].push_back(std::make_pair(c, val));
			}
		}   
 		posterior_writer.Write(key, utt_post);
   	num_done++;
  }

	KALDI_LOG << "Done copying " << num_done << " posteriors.";
	return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
	std::cerr << e.what();
	return -1;
  }
}
