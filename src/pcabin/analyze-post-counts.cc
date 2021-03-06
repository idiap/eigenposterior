// bin/analyze-counts.cc

// Copyright 2012-2016 Brno University of Technology (Author: Karel Vesely)
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

/** @brief Sums the pdf vectors to counts, this is used to obtain prior counts for hybrid decoding.
*/
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-pdf-prior.h"


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/posterior.h"
#include <iostream>
#include <fstream>
#include <queue>
#include <vector>

#include "base/timer.h"


#include <iomanip>
#include <algorithm>
#include <numeric>

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::uint64 uint64;
  try {
    const char *usage =
        "Computes element counts from post format features.\n"
        "This is meant for computing class counts from soft targets instead of hard targets.\n"
        "\n"
        "Usage: analyze-post-counts <post-rspecifier> <counts>\n";

    ParseOptions po(usage);

    bool binary = false;
    std::string symbol_table_filename = "";

    po.Register("binary", &binary, "write in binary mode");
    po.Register("symbol-table", &symbol_table_filename,
                "Read symbol table for display of counts");

    int32 counts_dim = 0;
    po.Register("counts-dim", &counts_dim,
                "Output dimension of the counts, "
                "a hint for dimension auto-detection.");

    std::string frame_weights;
    po.Register("frame-weights", &frame_weights,
                "Per-frame weights (counting weighted frames).");

    std::string utt_weights;
    po.Register("utt-weights", &utt_weights,
                "Per-utterance weights (counting weighted frames).");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string posteriors_rspecifier = po.GetArg(1),
        wxfilename = po.GetArg(2);

    SequentialPosteriorReader posterior_reader(posteriors_rspecifier);

    RandomAccessBaseFloatVectorReader weights_reader;
    if (frame_weights != "") {
      weights_reader.Open(frame_weights);
    }
    RandomAccessBaseFloatReader utt_weights_reader;
    if (utt_weights != "") {
      utt_weights_reader.Open(utt_weights);
    }

    // Buffer for accumulating the counts
    Vector<double> counts(counts_dim, kSetZero);

    int32 num_done = 0, num_other_error = 0;
    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      std::string utt = posterior_reader.Key();
			kaldi:: Posterior post = posterior_reader.Value();

      BaseFloat utt_w = 1.0;
      // Check if per-utterance weights are provided
      if (utt_weights != "") {
        if (!utt_weights_reader.HasKey(utt)) {
          KALDI_WARN << utt << ", missing per-utterance weight";
          num_other_error++;
          continue;
        } else {
          utt_w = utt_weights_reader.Value(utt);
        }
      }

      Vector<BaseFloat> frame_w;
      // Check if per-frame weights are provided
      if (frame_weights != "") {
        if (!weights_reader.HasKey(utt)) {
          KALDI_WARN << utt << ", missing per-frame weights";
          num_other_error++;
          continue;
        } else {
          frame_w = weights_reader.Value(utt);
        }
      }

      // Accumulate the counts
      for (size_t i = 0; i < post.size(); i++) {
				for (size_t j = 0; j < post[i].size(); j++) {
        	if (post[i][j].first >= counts.Dim()) {
	          counts.Resize(post[i][j].first+1, kCopyData);
	        }
  	      if (frame_weights != "") {
    	      counts(post[i][j].first) += post[i][j].second * utt_w * frame_w(i);
      	  } else {
        	  counts(post[i][j].first) += post[i][j].second * utt_w;
        	}
				}
      }
      num_done++;
    }

    // Report elements with zero counts
    for (size_t i = 0; i < counts.Dim(); i++) {
      if (0.0 == counts(i)) {
        KALDI_WARN << "Zero count for label " << i << ", this is suspicious.";
      }
    }

    // Add a ``half-frame'' to all the elements to
    // avoid zero-counts which would cause problems in decoding
    Vector<double> counts_nozero(counts);
    counts_nozero.Add(0.5);

    Output ko(wxfilename, binary);
    counts_nozero.Write(ko.Stream(), binary);

    //
    // THE REST IS FOR ANALYSIS, IT GETS PRINTED TO LOG
    //
    if (symbol_table_filename != "" || (kaldi::g_kaldi_verbose_level >= 1)) {
      // load the symbol table
      fst::SymbolTable *elem_syms = NULL;
      if (symbol_table_filename != "") {
          elem_syms = fst::SymbolTable::ReadText(symbol_table_filename);
          if (!elem_syms)
            KALDI_ERR << "Could not read symbol table from file "
                      << symbol_table_filename;
      }

      // sort the counts
      std::vector<std::pair<double, int32> > sorted_counts;
      for (int32 i = 0; i < counts.Dim(); i++) {
        sorted_counts.push_back(
                        std::make_pair(static_cast<double>(counts(i)), i));
      }
      std::sort(sorted_counts.begin(), sorted_counts.end());
      std::ostringstream os;
      double sum = counts.Sum();
      os << "Printing...\n### The sorted count table," << std::endl;
      os << "count\t(norm),\tid\t(symbol):" << std::endl;
      for (int32 i = 0; i < sorted_counts.size(); i++) {
        os << sorted_counts[i].first << "\t("
           << static_cast<float>(sorted_counts[i].first) / sum << "),\t"
           << sorted_counts[i].second << "\t"
           << (elem_syms != NULL ? "(" +
                           elem_syms->Find(sorted_counts[i].second) + ")" : "")
           << std::endl;
      }
      os << "\n#total " << sum
         << " (" << static_cast<float>(sum)/100/3600 << "h)"
         << std::endl;
      KALDI_LOG << os.str();
    }

    KALDI_LOG << "Summed " << num_done << " int32 vectors to counts, "
              << "skipped " << num_other_error << " vectors.";
    KALDI_LOG << "Counts written to " << wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
