// bin/est-pca.cc

// Copyright		2014	Johns Hopkins University	(author: Daniel Povey)
// Copyright		2017 Pranay Dighe (Idiap Research Institute)

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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"
#include "base/timer.h"

int main(int argc, char *argv[]) {
	using namespace kaldi;
	typedef kaldi::int32 int32;
	try {
	const char *usage =
		"Estimate PCA transform with mean of the data and information about how many components\n"
		"are required for n-percentile energy for n=1..100.\n"
		"Senone-id is required to be written as key in the output files."
		"This script is an extenstion of est-pca (see it for more details)\n"
		"\n"
		"Usage:	est-pca-mean-energy [options] (<feature-rspecifier>|<vector-rspecifier>) <pca-matrix-out> <mean-vector-out> <energy-int-vector-out>\n";

	ParseOptions po(usage);

	std::string senone;
	po.Register("senone-id", &senone, "Provide the senone-id");

	bool apply_log = false;
	po.Register("apply-log", &apply_log, "Transform input data to logscale. Might be needed if data is in probability domain.");

	bool binary = true;
	po.Register("binary", &binary, "Write accumulators in binary mode.");

	bool read_vectors = false;
	po.Register("read-vectors", &read_vectors, "If true, read in single vectors "
				"instead of feature matrices");

	bool normalize_variance = false;
	po.Register("normalize-variance", &normalize_variance, "If true, make a "
				"transform that normalizes variance to one.");

	bool normalize_mean = false;
	po.Register("normalize-mean", &normalize_mean, "If true, output an affine "
				"transform that subtracts the data mean.");

	int32 dim = -1;
	po.Register("dim", &dim, "Feature dimension requested (if <= 0, uses full "
				"feature dimension");

	std::string full_matrix_wxfilename;
	po.Register("write-full-matrix", &full_matrix_wxfilename, "Write full version of the matrix to this location (including rejected rows)");

	po.Read(argc, argv);

	if (po.NumArgs() != 4) {
		po.PrintUsage();
		exit(1);
	}

	Timer time;
	double time_now = 0;

	std::string rspecifier = po.GetArg(1),
				pca_wspecifier = po.GetArg(2),
				mean_wspecifier = po.GetArg(3),
				energy_wspecifier = po.GetArg(4);

	int32 num_done = 0, num_err = 0;
	int64 count = 0;
	Vector<double> sum;
	SpMatrix<double> sumsq;

	BaseFloatMatrixWriter pca_writer(pca_wspecifier);
	DoubleVectorWriter mean_writer(mean_wspecifier);
	Int32VectorWriter energy_writer(energy_wspecifier);
		
	float epsilon = 2.2204e-16;

	if (!read_vectors) {
		SequentialBaseFloatMatrixReader feat_reader(rspecifier);
			
		for (; !feat_reader.Done(); feat_reader.Next()) {
			Matrix<double> mat(feat_reader.Value());
			if (apply_log) {
				mat.ApplyFloor(epsilon);
				mat.ApplyLog();
			}
			if (mat.NumRows() == 0) {
				KALDI_WARN << "Empty feature matrix";
				num_err++;
				continue;
			}
			if (sum.Dim() == 0) {
				sum.Resize(mat.NumCols());
				sumsq.Resize(mat.NumCols());
			}
			if (sum.Dim() != mat.NumCols()) {
				KALDI_WARN << "Feature dimension mismatch " << sum.Dim() << " vs. "
						 << mat.NumCols();
				num_err++;
				continue;
			}
			sum.AddRowSumMat(1.0, mat);
			sumsq.AddMat2(1.0, mat, kTrans, 1.0);
			count += mat.NumRows();
			num_done++;
		}
		KALDI_LOG << "Accumulated stats from " << num_done << " feature files, "
				<< num_err << " with errors; " << count << " frames.";		
	} else {
		// read in vectors, not matrices
		SequentialBaseFloatVectorReader vec_reader(rspecifier);
	
		for (; !vec_reader.Done(); vec_reader.Next()) {
			Vector<double> vec(vec_reader.Value());
			if (vec.Dim() == 0) {
				KALDI_WARN << "Empty input vector";
				num_err++;
				continue;
			}
			if (sum.Dim() == 0) {
				sum.Resize(vec.Dim());
				sumsq.Resize(vec.Dim());
			}
			if (sum.Dim() != vec.Dim()) {
				KALDI_WARN << "Feature dimension mismatch " << sum.Dim() << " vs. "
						 << vec.Dim();
				num_err++;
				continue;
			}
			sum.AddVec(1.0, vec);
			sumsq.AddVec2(1.0, vec);
			count += 1.0;
			num_done++;
		}
		KALDI_LOG << "Accumulated stats from " << num_done << " vectors, "
				<< num_err << " with errors.";
	}
	if (num_done == 0)
		KALDI_ERR << "No data accumulated.";
	sum.Scale(1.0 / count);
	sumsq.Scale(1.0 / count);

	sumsq.AddVec2(-1.0, sum); // now sumsq is centered covariance.

	int32 full_dim = sum.Dim();
	if (dim <= 0) dim = full_dim;
	if (dim > full_dim)
		KALDI_ERR << "Final dimension " << dim << " is greater than feature "
				<< "dimension " << full_dim;
	
	Matrix<double> P(full_dim, full_dim);
	Vector<double> s(full_dim);
	
	sumsq.Eig(&s, &P);
	SortSvd(&s, &P);
	
	KALDI_VLOG(1) << "Sum of PCA eigenvalues is " << s.Sum() << ", sum of kept "
				<< "eigenvalues is " << s.Range(0, dim).Sum();

	Vector<BaseFloat> cumsum;
	cumsum.Resize(s.Dim());
	cumsum(0) = s(0);
	for ( int32 j = 1; j < cumsum.Dim(); j++) {
		cumsum(j) = cumsum(j-1) + s(j);
	}

	Vector<BaseFloat> energy;
	energy.Resize(s.Dim());
	energy.AddVec(100.0/s.Sum(),cumsum);

	std::vector<int32> e;
	e.resize(101);
	for( int32 i = 0; i <= 100; i++ ) {
		for ( int32 j = 0; j < energy.Dim(); j++) {
			if( energy(j) > i) {
				e[i] = j + 1;
				break;
			}
		}
	}
		
	Matrix<double> transform(P, kTrans); // Transpose of P.	This is what
										 // appears in the transform.
	if (normalize_variance) {
		for (int32 i = 0; i < full_dim; i++) {
			double this_var = s(i), min_var = 1.0e-15;
			if (this_var < min_var) {
				KALDI_WARN << "--normalize-variance option: very tiny variance " << s(i)
						 << "encountered, treating as " << min_var;
				this_var = min_var;
			}
			double scale = 1.0 / sqrt(this_var); // scale on features that will make
												 // the variance unit.
			transform.Row(i).Scale(scale);
		}
	}

	Vector<double> offset(full_dim);
	
	if (normalize_mean) {
		offset.AddMatVec(-1.0, transform, kNoTrans, sum, 0.0);
		transform.Resize(full_dim, full_dim + 1, kCopyData); // Add column to transform.
		transform.CopyColFromVec(offset, full_dim);
	}

	Matrix<BaseFloat> transform_float(transform);

	Matrix<BaseFloat> pca_mat(transform_float, kTrans); //Final Matrix to be written.

	//Only store as many Principal Components as required for 0 to 99 percentile energy conversation.
	SubMatrix<BaseFloat> trans_pca(pca_mat, 0, full_dim,0,(int32) e[99]);
	Matrix<BaseFloat> out;
	out.Resize(trans_pca.NumRows(), trans_pca.NumCols());
	out = trans_pca;

	Vector<double> mean(sum);

	const std::string &senone_id = senone;
	pca_writer.Write(senone_id, out);
	mean_writer.Write(senone_id, mean);

	const std::vector<int32> bins(e);
	energy_writer.Write(senone_id, bins);

	time_now = time.Elapsed();
	KALDI_LOG << "Time Elapsed is " << time_now/60 << "mins.";

	return 0;
	} catch(const std::exception &e) {
	std::cerr << e.what();
	return -1;
	}
}
