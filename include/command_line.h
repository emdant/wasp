// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef COMMAND_LINE_H_
#define COMMAND_LINE_H_

#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <CLI11.hpp>
#include <utility>

enum class GraphGenerator : int {
  NO_GEN,
  KRONECKER,
  UNIFORM
};

enum class WeightGenerator : int {
  NO_GEN,
  UNIFORM,
  NORMAL,
};

enum OutputFormat : int {
  GAP_BINARY,
  EDGE_LIST,
  MATRIX_MARKET
};

class CLBase {
private:
  // clang-format off
  std::map<std::string, GraphGenerator> graph_map_{
    {"kron", GraphGenerator::KRONECKER}, {"kronecker", GraphGenerator::KRONECKER},
    {"uni", GraphGenerator::UNIFORM}, {"uniform", GraphGenerator::UNIFORM}
  };

  std::map<std::string, WeightGenerator> weight_map_{
    {"normal", WeightGenerator::NORMAL}, {"uniform", WeightGenerator::UNIFORM}
  };

  // clang-format on

  void print_options(const std::vector<CLI::Option*>& options) {
    auto join_strings = [](const std::vector<std::string>& vec, const std::string& delim) {
      if (vec.empty()) {
        return std::string("");
      }
      // Use std::accumulate to efficiently join the strings
      return std::accumulate(
          std::next(vec.begin()),
          vec.end(),
          vec[0],
          [&delim](const std::string& a, const std::string& b) {
            return a + delim + b;
          }
      );
    };

    // Iterate through all options
    for (const CLI::Option* opt : options) {
      // Skip the help flag
      if (opt->get_name() == "--help") {
        continue;
      }

      std::cout << opt->get_name() << ": ";

      // Check if the option was passed on the command line
      if (opt->count() > 0) {
        if (opt->get_expected_min() == 0) std::cout << "true [CLI]";
        else std::cout << join_strings(opt->results(), ", ") << " [CLI]";
      } else if (!opt->get_default_str().empty()) {
        if (opt->get_expected_min() == 0) std::cout << (opt->get_default_str() == "0" ? "false" : "true") << " [default]";
        else std::cout << opt->get_default_str() << " [default]"; // If not passed, print its default value
      } else {
        std::cout << "[not set]"; // For options without a default that were not passed
      }
      std::cout << std::endl;
    }
  }

  void print_config(CLI::App& app) {
    std::cout << "------ Configuration ------" << std::endl;

    // Determine which modes are active based on user input.
    const bool is_file_mode = app.get_option("--graph-filename")->count() > 0;
    const bool is_synthetic_mode = app.get_option("--synthetic-gen")->count() > 0;
    const bool is_weight_gen_mode = app.get_option("--weight-gen")->count() > 0;
    // Use get_option_no_throw for options from derived classes that may not exist in CLBase.
    const CLI::Option* sources_file_opt = app.get_option_no_throw("--sources");
    const bool is_sources_file_mode = sources_file_opt && sources_file_opt->count() > 0;

    std::vector<CLI::Option*> options_to_print;
    for (CLI::Option* opt : app.get_options()) {
      // Basic filtering for help flags.
      if (!opt || opt->get_name() == "--help" || opt->get_name() == "?") {
        continue;
      }

      const std::string& opt_name = opt->get_name();

      // Rule 1: Exclude options based on the primary graph mode (file vs. synthetic).
      if (is_file_mode && (opt_name == "--synthetic-gen" || opt_name == "--scale" || opt_name == "--degree"))
        continue;

      if (is_synthetic_mode && (opt_name == "--graph-filename" || opt_name == "--weights-filename" || opt_name == "--symmetrize"))
        continue;

      // Rule 2: Exclude weight options if not generating weights.
      if (opt_name == "--weight-range" && !is_weight_gen_mode)
        continue;

      if (opt_name == "--weight-gen" && !is_weight_gen_mode)
        continue;

      // Rule 4: Exclude conflicting traversal options if a sources file is provided.
      if (is_sources_file_mode && opt_name == "--start-vertex")
        continue;

      // If we reach here, the option is relevant to the current mode.
      if (opt->count() > 0 || opt->get_expected_min() == 0 || !opt->get_default_str().empty())
        options_to_print.push_back(opt);
    }

    print_options(options_to_print);
    std::cout << "---------------------" << std::endl;
  }

protected:
  CLI::App app_;
  int argc_;
  char** argv_;

  // Member variables to hold the parsed option values
  std::string graph_filename_{""};
  std::string weights_filename_{""};
  bool symmetrize_{false};
  GraphGenerator gen_{GraphGenerator::NO_GEN};
  int gen_scale_{16};
  int gen_degree_{16};

  WeightGenerator weight_dist_{WeightGenerator::NO_GEN};
  std::pair<double, double> weight_range_;

public:
  explicit CLBase(int argc, char** argv, std::string name)
      : app_(name), argc_(argc), argv_(argv) {
    auto gfname = app_.add_option("-f,--graph-filename", graph_filename_, "Load graph from file")
                      ->default_val("");

    app_.add_option("-w,--weights-filename", weights_filename_, "Load weights from file")
        ->default_val("")
        ->needs(gfname);

    app_.add_flag("--symmetrize", symmetrize_, "Symmetrize input edge list")
        ->default_val(false);

    auto synthetic_gen = app_.add_option("--synthetic-gen", gen_, "Kind of synthetic graph to generate")
                             ->transform(CLI::CheckedTransformer(graph_map_, CLI::ignore_case));
    app_.add_option("--scale", gen_scale_, "Scale of the synthetic graph (2^{scale})")
        ->needs(synthetic_gen)
        ->default_val(16);

    app_.add_option("--degree", gen_degree_, "Average degree of the  synthetic graph")
        ->needs(synthetic_gen)
        ->default_val(16);

    auto wt = app_.add_option("--weight-gen", weight_dist_, "Kind of synthetic weights to generate")
                  ->default_val(WeightGenerator::NO_GEN)
                  ->transform(CLI::CheckedTransformer(weight_map_, CLI::ignore_case));

    app_.add_option("--weight-range", weight_range_, "Range [a, b) of the uniform distribution")
        ->needs(wt)
        ->expected(0, 2);

    synthetic_gen->needs(wt);

    gfname->excludes(synthetic_gen);
    synthetic_gen->excludes(gfname);
  }

  virtual ~CLBase() = default;

  void parse() {
    try {
      app_.parse(argc_, argv_);
    } catch (const CLI::ParseError& e) {
      std::exit(app_.exit(e));
    }
    print_config(app_);
  }

  std::string graph_filename() const { return graph_filename_; }
  std::string weights_filename() const { return weights_filename_; }
  bool symmetrize() const { return symmetrize_; }
  GraphGenerator graph_generator() const { return gen_; }
  int synthetic_scale() const { return gen_scale_; }
  int synthetic_degree() const { return gen_degree_; }
  bool using_generator() const { return gen_ != GraphGenerator::NO_GEN; }

  WeightGenerator weight_distribution() const { return weight_dist_; }
  std::pair<double, double> weight_range() const { return weight_range_; }
};

class CLApp : public CLBase {
protected:
  int num_trials_{12};
  bool analysis_{false};
  bool verify_{false};

public:
  explicit CLApp(int argc, char** argv, std::string name) : CLBase(argc, argv, name) {
    app_.add_option("-n,--num-trials", num_trials_, "Number of trials to perform")->default_val(12);
    app_.add_flag("-a,--analysis", analysis_, "Output analysis of last run")->default_val(false);
    app_.add_flag("-v,--verify", verify_, "Verify the output of each run")->default_val(false);
  }

  int num_trials() const { return num_trials_; }
  bool analysis() const { return analysis_; }
  bool verify() const { return verify_; }
};

class CLTraversal : public CLApp {
protected:
  std::string sources_file_;
  int64_t start_vertex_{-1};
  int num_sources_{1};

public:
  explicit CLTraversal(int argc, char** argv, std::string name) : CLApp(argc, argv, name) {
    auto sf = app_.add_option("-s,--sources", sources_file_, "Load sources from file")
                  ->default_val("");
    auto sv = app_.add_option("-r,--start-vertex", start_vertex_, "Start traversal from vertex")
                  ->default_val(-1) // -1 indicates not set by user
                  ->default_str("randomly generated");
    app_.add_option("-S,--num-sources", num_sources_, "Number of source vertices to test")
        ->default_val(1);

    sf->excludes(sv);
    sv->excludes(sf);
  }

  std::string sources_filename() const { return sources_file_; }
  int64_t start_vertex() const { return start_vertex_; }
  int num_sources() const { return num_sources_; }
  bool start_vertex_is_set() const {
    return app_.count("--start-vertex") > 0;
  }
};

template <typename WeightT_>
class CLDelta : public CLTraversal {
protected:
  WeightT_ delta_{1};

public:
  explicit CLDelta(int argc, char** argv, std::string name)
      : CLTraversal(argc, argv, name) {
    app_.add_option("-d,--delta", delta_, "Delta parameter")->default_val(1);
  }

  WeightT_ delta() const { return delta_; }
};

class CLConverter : public CLBase {
private:
  // clang-format off
  std::map<std::string, OutputFormat> format_map_{
    {"gap", OutputFormat::GAP_BINARY}, {"el", OutputFormat::EDGE_LIST}, {"mtx", OutputFormat::MATRIX_MARKET}
  };
  // clang-format on

protected:
  std::string out_filename_;
  OutputFormat out_format_;
  bool out_weighted_{false};
  bool relabel_vertices_{false};

public:
  explicit CLConverter(int argc, char** argv, std::string name)
      : CLBase(argc, argv, name) {
    app_.add_option("-o,--output", out_filename_, "Output graph to this file")
        ->required();
    app_.add_option("--format", out_format_, "Output format")
        ->required()
        ->transform(CLI::CheckedTransformer(format_map_, CLI::ignore_case));

    app_.add_flag("--weighted", out_weighted_, "Make output weighted")
        ->default_val(false);
    app_.add_flag("--relabel", relabel_vertices_, "Relabel vertices from 0 to |V|-1")
        ->default_val(false);
  }

  std::string out_filename() const { return out_filename_; }
  OutputFormat out_format() const { return out_format_; }
  bool out_weighted() const { return out_weighted_; }
  bool relabel_vertices() const { return relabel_vertices_; }
};

class CLExport : public CLBase {
protected:
  std::string out_filename_;

public:
  explicit CLExport(int argc, char** argv, std::string name)
      : CLBase(argc, argv, name) {
    app_.add_option("-o,--output", out_filename_, "Output exported data to this file")
        ->required();
  }

  std::string out_filename() const { return out_filename_; }
};

class CLStats : public CLBase {
protected:
  std::string out_directory_;

public:
  explicit CLStats(int argc, char** argv, std::string name)
      : CLBase(argc, argv, name) {
    app_.add_option("-d,--output-directory", out_directory_, "Output directory")
        ->required();
  }

  std::string out_directory() const { return out_directory_; }
};

class CLEdge : public CLBase {
protected:
  std::int32_t src, dst;

public:
  explicit CLEdge(int argc, char* argv[], std::string name) : CLBase(argc, argv, name) {
    app_.add_option("src", src, "Edge source")->check(CLI::Number)->required();
    app_.add_option("dst", dst, "Edge destination")->check(CLI::Number)->required();
    app_.validate_positionals();
  }

  int32_t source() const { return src; }
  int32_t destination() const { return dst; }
};

#endif // COMMAND_LINE_H_