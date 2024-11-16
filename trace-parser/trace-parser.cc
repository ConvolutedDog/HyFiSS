#include <bits/stdc++.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <time.h>
#include <vector>

#include "trace-parser.h"

void app_config::init(std::string config_path, bool PRINT_LOG) {

  std::stringstream ss;

  std::ifstream inputFile;
  inputFile.open(config_path);
  if (!inputFile.good()) {
    fprintf(stderr, "\n\nOptionParser ** ERROR: Cannot open config file '%s'\n",
            config_path.c_str());
    exit(1);
  }

  std::string target_app_kernels_id = "-app_kernels_id";
  std::string target_concurrentKernels = "-device_concurrentKernels";

  std::string line;
  size_t commentStart;
  size_t found;
  std::string result;
  int comma_count;
  while (inputFile.good()) {
    getline(inputFile, line);

    commentStart = line.find_first_of("#");
    if (commentStart != line.npos)
      continue;
    found = line.find(target_app_kernels_id);

    if (found != std::string::npos) {
      result = line.substr(found + target_app_kernels_id.length());
      comma_count = std::count(result.begin(), result.end(), ',');
      kernels_num = comma_count + 1;

      app_kernels_id_string =
          line.substr(found + target_app_kernels_id.length() + 1);
    }
    found = line.find(target_concurrentKernels);
    if (found != std::string::npos) {
      result = line.substr(found + target_concurrentKernels.length());

      concurrentKernels = std::stoi(result);
    }
  }
  inputFile.clear();
  inputFile.seekg(0, std::ios::beg);

  kernel_name.resize(kernels_num);
  kernel_num_registers.resize(kernels_num);
  kernel_shared_mem_bytes.resize(kernels_num);
  kernel_grid_size.resize(kernels_num);
  kernel_block_size.resize(kernels_num);
  kernel_cuda_stream_id.resize(kernels_num);

  kernel_grid_dim_x.resize(kernels_num);
  kernel_grid_dim_y.resize(kernels_num);
  kernel_grid_dim_z.resize(kernels_num);
  kernel_tb_dim_x.resize(kernels_num);
  kernel_tb_dim_y.resize(kernels_num);
  kernel_tb_dim_z.resize(kernels_num);
  kernel_shmem_base_addr.resize(kernels_num);
  kernel_local_base_addr.resize(kernels_num);

#ifdef ENABLE_SAMPLING_POINT
  kernel_sampling_point.resize(kernels_num);
#endif

  for (int j = 0; j < kernels_num; ++j) {
    while (inputFile.good()) {
      getline(inputFile, line);
      commentStart = line.find_first_of("#");
      if (commentStart != line.npos)
        continue;

      ss.str("");
      ss << "-kernel_" << j + 1 << "_kernel_name";
      found = line.find(ss.str());
      if (found != std::string::npos) {
        kernel_name[j] = line.substr(found + ss.str().length());
      }

      ss.str("");
      ss << "-kernel_" << j + 1 << "_num_registers";
      found = line.find(ss.str());
      if (found != std::string::npos) {
        kernel_num_registers[j] =
            std::stoi(line.substr(found + ss.str().length()));
      }

      ss.str("");
      ss << "-kernel_" << j + 1 << "_shared_mem_bytes";
      found = line.find(ss.str());
      if (found != std::string::npos) {
        kernel_shared_mem_bytes[j] =
            std::stoi(line.substr(found + ss.str().length()));
      }

      ss.str("");
      ss << "-kernel_" << j + 1 << "_grid_size";
      found = line.find(ss.str());
      if (found != std::string::npos) {
        kernel_grid_size[j] = std::stoi(line.substr(found + ss.str().length()));
      }

      ss.str("");
      ss << "-kernel_" << j + 1 << "_block_size";
      found = line.find(ss.str());
      if (found != std::string::npos) {
        kernel_block_size[j] =
            std::stoi(line.substr(found + ss.str().length()));
      }

      ss.str("");
      ss << "-kernel_" << j + 1 << "_cuda_stream_id";
      found = line.find(ss.str());
      if (found != std::string::npos) {
        kernel_cuda_stream_id[j] =
            std::stoi(line.substr(found + ss.str().length()));
      }

      ss.str("");
      ss << "-kernel_" << j + 1 << "_grid_dim_x";
      found = line.find(ss.str());
      if (found != std::string::npos) {
        kernel_grid_dim_x[j] =
            std::stoi(line.substr(found + ss.str().length()));
      }

      ss.str("");
      ss << "-kernel_" << j + 1 << "_grid_dim_y";
      found = line.find(ss.str());
      if (found != std::string::npos) {
        kernel_grid_dim_y[j] =
            std::stoi(line.substr(found + ss.str().length()));
      }

      ss.str("");
      ss << "-kernel_" << j + 1 << "_grid_dim_z";
      found = line.find(ss.str());
      if (found != std::string::npos) {
        kernel_grid_dim_z[j] =
            std::stoi(line.substr(found + ss.str().length()));
      }

      ss.str("");
      ss << "-kernel_" << j + 1 << "_tb_dim_x";
      found = line.find(ss.str());
      if (found != std::string::npos) {
        kernel_tb_dim_x[j] = std::stoi(line.substr(found + ss.str().length()));
      }

      ss.str("");
      ss << "-kernel_" << j + 1 << "_tb_dim_y";
      found = line.find(ss.str());
      if (found != std::string::npos) {
        kernel_tb_dim_y[j] = std::stoi(line.substr(found + ss.str().length()));
      }

      ss.str("");
      ss << "-kernel_" << j + 1 << "_tb_dim_z";
      found = line.find(ss.str());
      if (found != std::string::npos) {
        kernel_tb_dim_z[j] = std::stoi(line.substr(found + ss.str().length()));
      }

      ss.str("");
      ss << "-kernel_" << j + 1 << "_shmem_base_addr";
      found = line.find(ss.str());
      if (found != std::string::npos) {
        kernel_shmem_base_addr[j] =
            std::stoull(line.substr(found + ss.str().length()), 0, 16);
      }

      ss.str("");
      ss << "-kernel_" << j + 1 << "_local_base_addr";
      found = line.find(ss.str());
      if (found != std::string::npos) {
        kernel_local_base_addr[j] =
            std::stoull(line.substr(found + ss.str().length()), 0, 16);
      }

#ifdef ENABLE_SAMPLING_POINT
      ss.str("");
      ss << "-kernel_" << j + 1 << "_sampling_point";
      found = line.find(ss.str());
      if (found != std::string::npos) {
        kernel_sampling_point[j] = std::stoi(line.substr(found + ss.str().length()));
      }
#endif      
    }
    inputFile.clear();
    inputFile.seekg(0, std::ios::beg);
  }

  inputFile.close();

  char *toks = new char[2048];
  char *tokd = toks;
  strcpy(toks, app_kernels_id_string.c_str());

  app_kernels_id.resize(kernels_num);

  toks = strtok(toks, ",");

  for (int i = 0; i < kernels_num; i++) {
    assert(toks);
    int ntok = sscanf(toks, "%d", &app_kernels_id[i]);
    assert(ntok == 1);
    toks = strtok(NULL, ",");
  }

  delete[] tokd;
  delete[] toks;

  if (PRINT_LOG)
    fprintf(stdout, ">>> APP config Options <<<:\n");

  m_valid = true;
}

void instn_config::init(std::string config_path, bool PRINT_LOG) {
  std::ifstream inputFile;

  inputFile.open(config_path);
  if (!inputFile.good()) {
    fprintf(stderr, "\n\nOptionParser ** ERROR: Cannot open config file '%s'\n",
            config_path.c_str());
    exit(1);
  }

  size_t first_blank_pos;
  size_t second_blank_pos;
  unsigned kernel_id, pc;
  std::string kernel_id_str, pc_str, instn_str;

  while (inputFile.good()) {
    std::string line;
    getline(inputFile, line);
    size_t commentStart = line.find_first_of("#");
    if (commentStart != line.npos)
      continue;
    if (!line.empty()) {
      first_blank_pos = line.find(' ');
      second_blank_pos = line.find(' ', first_blank_pos + 1);
      kernel_id_str = line.substr(0, first_blank_pos);
      pc_str = line.substr(first_blank_pos + 1,
                           second_blank_pos - first_blank_pos - 1);

      std::istringstream iss_1(kernel_id_str);
      std::istringstream iss_2(pc_str);

      iss_1 >> kernel_id;
      iss_2 >> std::hex >> pc;

      instn_str = line.substr(second_blank_pos + 1);

      _inst_trace_t *instn_info =
          new _inst_trace_t(kernel_id - 1, pc, instn_str, hw_cfg);

      instn_info_vector[std::make_pair(kernel_id - 1, pc)] = instn_info;
    }
  }
  inputFile.close();

  if (PRINT_LOG)
    fprintf(stdout, ">>> INSTN config Options <<<:\n");
}

int issue_config::get_sm_id_of_one_block(unsigned kernel_id,
                                         unsigned block_id) {

  for (unsigned i = 0; i < trace_issued_sm_id_blocks.size(); i++) {
    for (unsigned j = 0; j < trace_issued_sm_id_blocks[i].size(); j++) {

      if (trace_issued_sm_id_blocks[i][j].kernel_id == kernel_id &&
          trace_issued_sm_id_blocks[i][j].block_id == block_id) {
        return trace_issued_sm_id_blocks[i][j].sm_id;
      }
    }
  }
  return -1;
}

int issue_config::get_sm_id_of_one_block_fast(unsigned kernel_id,
                                              unsigned block_id) {
  return trace_issued_sm_id_blocks_map[std::make_pair(kernel_id, block_id)];
}

/// @brief Parse the thread block info from the string.
/// @param blocks_info_str The string indicates the thread blocks that
///        are being emitted to an SM.
/// @return A vector of type `std::vector<block_info_t>`.
std::vector<block_info_t> issue_config::parse_blocks_info(
  const std::string &blocks_info_str) {
  std::vector<block_info_t> result;
  size_t start = 0;
  size_t end = blocks_info_str.find(',', start);
  int total_tuples = std::stoi(blocks_info_str.substr(start, end - start));

  start = end + 1;
  end = blocks_info_str.find(',', end + 1);
  int sm_id = std::stoi(blocks_info_str.substr(start, end - start));

  for (int i = 0; i < total_tuples; ++i) {
    start = end + 1;
    end = blocks_info_str.find('(', start);
    size_t comma = blocks_info_str.find(',', end);

    unsigned kernel_id =
        std::stoi(blocks_info_str.substr(end + 1, comma - end - 1));

    end = blocks_info_str.find(',', comma + 1);

    unsigned block_id =
        std::stoi(blocks_info_str.substr(comma + 1, end - comma - 1));

    comma = blocks_info_str.find(')', end + 1);

    unsigned long long time_stamp =
        std::stoull(blocks_info_str.substr(end + 1, comma - end - 1), 0, 16);

    end = comma + 1;

    trace_issued_sm_id_blocks_map[std::make_pair(kernel_id, block_id)] = sm_id;

    block_info_t info = block_info_t(kernel_id, block_id, time_stamp, sm_id);

    result.push_back(info);
  }

  return result;
}

void issue_config::init(const std::string config_path, bool dump_log) {
  std::stringstream ss;

  std::ifstream inputFile;
  inputFile.open(config_path);
  if (!inputFile.good()) {
    fprintf(stderr, "\n\nERROR: Cannot open issue.config file '%s'\n",
            config_path.c_str());
    exit(1);
  }

  // Target of `trace_issued_sms_num` means the number of SMs that the
  // application emits thread blocks to.
  std::string target1 = "-trace_issued_sms_num";
  // Target of `trace_issued_sms_vector` means the indexes of SMs that
  // the application emits thread blocks to.
  std::string target2 = "-trace_issued_sms_vector";
  std::string line;
  size_t commentStart;
  size_t found1, found2;
  std::string result1, result2;
  while (inputFile.good()) {
    getline(inputFile, line);
    commentStart = line.find_first_of("#");
    if (commentStart != line.npos)
      continue;
    found1 = line.find(target1);
    found2 = line.find(target2);
    if (found1 != std::string::npos) {
      result1 = line.substr(found1 + target1.length() + 1);
      // The number of SMs that the application emits thread blocks to.
      trace_issued_sms_num = std::stoi(result1);
    }
    if (found2 != std::string::npos) {
      result2 = line.substr(found2 + target2.length() + 1);

      std::istringstream iss(result2);
      std::string token;
      // The indexes of SMs that the application emits thread blocks to.
      while (std::getline(iss, token, ',')) {
        trace_issued_sms_vector.push_back(std::stoi(token));
      }
    }
  }
  inputFile.clear();
  inputFile.seekg(0, std::ios::beg);

  // An object of type `vector<string>`. It stores the string of thread
  // blocks that are emitted to each SM.
  trace_issued_sm_id_blocks_str.resize(trace_issued_sms_num);

  std::vector<int> has_found_j;
  while (inputFile.good()) {
    getline(inputFile, line);
    commentStart = line.find_first_of("#");
    if (commentStart != line.npos)
      continue;

    // There is such a scenario that the SMs to which the thread blocks
    // are emitted may not be indexed in the order from 0 to 1 to 2, for
    // example, it may be that SM-0, SM-2, and SM-4 to which the thread
    // blocks are emitted to.
    for (int j = 0; j < trace_issued_sms_num; ++j) {
      /// TODO: This loop here need to be replaced, cause none-loop can
      /// also impletement this function.
      if (std::find(has_found_j.begin(), has_found_j.end(), j) ==
          has_found_j.end()) {
        int sm_index = trace_issued_sms_vector[j];
        ss.str("");
        ss << "-trace_issued_sm_id_" << std::to_string(sm_index);
        size_t found = line.find(ss.str());
        if (found != std::string::npos) {
          has_found_j.push_back(j);
          trace_issued_sm_id_blocks_str[j] =
              line.substr(found + ss.str().length() + 1);
          break;
        }
      }
    }
  }
  inputFile.close();

  if (dump_log)
    fprintf(stdout, ">>> ISSUE config Options <<<:\n");

  std::string blocks_info_str;
  trace_issued_sm_id_blocks.resize(trace_issued_sms_num);
  for (int j = 0; j < trace_issued_sms_num; ++j) {
    blocks_info_str = trace_issued_sm_id_blocks_str[j].c_str();
    // `parse_blocks_info` is used to parse the thread block info from
    // the string. Each element in `trace_issued_sm_id_blocks` of type
    // `vector<...>` is a vector of type `vector<block_info_t>`.
    trace_issued_sm_id_blocks[j] = parse_blocks_info(blocks_info_str);
  }

  m_valid = true;
}

kernel_trace_t *trace_parser::parse_kernel_info(int kernel_id, bool PRINT_LOG) {
  kernel_trace_t *kernel_info = new kernel_trace_t;

  kernel_info->kernel_name = get_appcfg()->get_kernel_name(kernel_id);

  kernel_info->kernel_id =
      static_cast<unsigned>(get_appcfg()->get_app_kernel_id(kernel_id));
  kernel_info->grid_dim_x =
      static_cast<unsigned>(get_appcfg()->get_kernel_grid_dim_x(kernel_id));
  kernel_info->grid_dim_y =
      static_cast<unsigned>(get_appcfg()->get_kernel_grid_dim_y(kernel_id));
  kernel_info->grid_dim_z =
      static_cast<unsigned>(get_appcfg()->get_kernel_grid_dim_z(kernel_id));
  kernel_info->tb_dim_x =
      static_cast<unsigned>(get_appcfg()->get_kernel_tb_dim_x(kernel_id));
  kernel_info->tb_dim_y =
      static_cast<unsigned>(get_appcfg()->get_kernel_tb_dim_y(kernel_id));
  kernel_info->nregs =
      static_cast<unsigned>(get_appcfg()->get_kernel_num_registers(kernel_id));
  kernel_info->cuda_stream_id =
      static_cast<unsigned>(get_appcfg()->get_kernel_cuda_stream_id(kernel_id));
  kernel_info->shmem = static_cast<unsigned>(
      get_appcfg()->get_kernel_shmem_base_addr(kernel_id));

  kernel_info->binary_verion = VOLTA_BINART_VERSION;

  kernel_info->enable_lineinfo = 0;

  kernel_info->nvbit_verion = "1.5.0";
  kernel_info->trace_verion = 0;
  kernel_info->shmem_base_addr =
      get_appcfg()->get_kernel_shmem_base_addr(kernel_id);
  kernel_info->local_base_addr =
      get_appcfg()->get_kernel_local_base_addr(kernel_id);

#ifdef ENABLE_SAMPLING_POINT
  kernel_info->sampling_point = 
      get_appcfg()->get_kernel_sampling_point(kernel_id);
#endif

  return kernel_info;
}

kernel_trace_t *
trace_parser::parse_kernel_info(const std::string &kerneltraces_filepath) {
  kernel_trace_t *kernel_info = new kernel_trace_t;
  kernel_info->enable_lineinfo = 0;
  kernel_info->ifs = new std::ifstream;
  std::ifstream *ifs = kernel_info->ifs;
  ifs->open(kerneltraces_filepath.c_str());

  if (!ifs->is_open()) {
    std::cout << "Unable to open file: " << kerneltraces_filepath << std::endl;
    exit(1);
  }

  std::cout << "Processing kernel " << kerneltraces_filepath << std::endl;

  std::string line;

  while (!ifs->eof()) {
    getline(*ifs, line);

    if (line.length() == 0) {
      continue;
    } else if (line[0] == '#') {

      break;
    } else if (line[0] == '-') {
      std::stringstream ss;
      std::string string1, string2;

      ss.str(line);
      ss.ignore();
      ss >> string1 >> string2;

      if (string1 == "kernel" && string2 == "name") {
        const size_t equal_idx = line.find('=');
        kernel_info->kernel_name = line.substr(equal_idx + 2);
      } else if (string1 == "kernel" && string2 == "id") {
        sscanf(line.c_str(), "-kernel id = %u", &kernel_info->kernel_id);
      } else if (string1 == "grid" && string2 == "dim") {
        sscanf(line.c_str(), "-grid dim = (%u,%u,%u)", &kernel_info->grid_dim_x,
               &kernel_info->grid_dim_y, &kernel_info->grid_dim_z);
      } else if (string1 == "block" && string2 == "dim") {
        sscanf(line.c_str(), "-block dim = (%u,%u,%u)", &kernel_info->tb_dim_x,
               &kernel_info->tb_dim_y, &kernel_info->tb_dim_z);
      } else if (string1 == "shmem" && string2 == "=") {
        sscanf(line.c_str(), "-shmem = %u", &kernel_info->shmem);
      } else if (string1 == "nregs") {
        sscanf(line.c_str(), "-nregs = %u", &kernel_info->nregs);
      } else if (string1 == "cuda" && string2 == "stream") {
        sscanf(line.c_str(), "-cuda stream id = %lu",
               &kernel_info->cuda_stream_id);
      } else if (string1 == "binary" && string2 == "version") {
        sscanf(line.c_str(), "-binary version = %u",
               &kernel_info->binary_verion);
      } else if (string1 == "enable" && string2 == "lineinfo") {
        sscanf(line.c_str(), "-enable lineinfo = %u",
               &kernel_info->enable_lineinfo);
      } else if (string1 == "nvbit" && string2 == "version") {
        const size_t equal_idx = line.find('=');
        kernel_info->nvbit_verion = line.substr(equal_idx + 1);

      } else if (string1 == "accelsim" && string2 == "tracer") {
        sscanf(line.c_str(), "-accelsim tracer version = %u",
               &kernel_info->trace_verion);

      } else if (string1 == "shmem" && string2 == "base_addr") {
        const size_t equal_idx = line.find('=');
        ss.str(line.substr(equal_idx + 1));
        ss >> std::hex >> kernel_info->shmem_base_addr;

      } else if (string1 == "local" && string2 == "mem") {
        const size_t equal_idx = line.find('=');
        ss.str(line.substr(equal_idx + 1));
        ss >> std::hex >> kernel_info->local_base_addr;
      }
      std::cout << "    Info: " << line << std::endl;
      continue;
    }
  }

  return kernel_info;
}

void trace_parser::kernel_finalizer(kernel_trace_t *trace_info) {
  assert(trace_info);
  assert(trace_info->ifs);
  if (trace_info->ifs->is_open())
    trace_info->ifs->close();
  delete trace_info->ifs;
  delete trace_info;
}

trace_parser::trace_parser(const char *input_configs_filepath) {
  configs_filepath = input_configs_filepath;
  m_valid = true;
}

#include <limits.h>
#include <unistd.h>

void trace_parser::process_configs_file(const std::string config_path,
                                        int config_type, bool PRINT_LOG) {
  std::ifstream fs;

  char cwd[PATH_MAX];
  assert(getcwd(cwd, sizeof(cwd)) != nullptr);
  std::string current_directory(cwd);

  std::string abs_config_path = current_directory + "/" + config_path;
  fs.open(abs_config_path);

  if (!fs.is_open()) {
    std::cout << "Unable to open file: " << abs_config_path << std::endl;
    exit(1);
  }
  fs.close();

  if (config_type == APP_CONFIG) {
    appcfg = app_config();
    appcfg.init(abs_config_path, PRINT_LOG);
  } else if (config_type == INSTN_CONFIG) {

    instncfg = instn_config(hw_cfg);

    instncfg.init(abs_config_path, PRINT_LOG);
  } else if (config_type == ISSUE_CONFIG) {
    issuecfg = issue_config();
    issuecfg.init(abs_config_path, PRINT_LOG);
  }
}

void trace_parser::judge_concurrent_issue() {}

void trace_parser::parse_configs_file(bool PRINT_LOG) {

  if (configs_filepath.back() == '/') {
    app_config_path = configs_filepath + "app.config";
    instn_config_path = configs_filepath + "instn.config";
    issue_config_path = configs_filepath + "issue.config";

  } else {
    app_config_path = configs_filepath + "/" + "app.config";
    instn_config_path = configs_filepath + "/" + "instn.config";
    issue_config_path = configs_filepath + "/" + "issue.config";
  }

  process_configs_file(app_config_path, APP_CONFIG, PRINT_LOG);

  process_configs_file(instn_config_path, INSTN_CONFIG, PRINT_LOG);

  process_configs_file(issue_config_path, ISSUE_CONFIG, PRINT_LOG);

  judge_concurrent_issue();
}

#include <algorithm>
#include <dirent.h>
#include <regex>

bool judge_format_mem(char *d_name, std::vector<std::pair<int, int>> *x) {
  std::string name = d_name;

  size_t pos_kernel = name.find("kernel_");
  size_t pos_block = name.find("_block_");
  size_t pos_mem = name.find(".mem");

  if (pos_kernel != std::string::npos && pos_block != std::string::npos &&
      pos_mem != std::string::npos) {
    int xx =
        std::stoi(name.substr(pos_kernel + 7, pos_block - (pos_kernel + 7)));
    int yy = std::stoi(name.substr(pos_block + 7, pos_mem - (pos_block + 7)));

    if (std::find((*x).begin(), (*x).end(), std::make_pair(xx, yy)) !=
        (*x).end()) {
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

bool judge_format_compute_kernel_id(char *d_name,
                                    std::vector<std::pair<int, int>> *x) {
  std::string name = d_name;
  std::regex pattern("kernel_(\\d+).sass");
  std::smatch match;

  if (std::regex_match(name, match, pattern)) {
    int xx = std::stoi(match[1]);

    std::vector<int> need_processed_kernel_ids;
    for (auto i : (*x)) {
      need_processed_kernel_ids.push_back(i.first);
    }

    if (std::find(need_processed_kernel_ids.begin(),
                  need_processed_kernel_ids.end(),
                  xx) != need_processed_kernel_ids.end()) {
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

bool judge_format_compute_kernel_id_fast(int kernel_id, int block_id,
                                         std::vector<std::pair<int, int>> *x) {

  if (std::find((*x).begin(), (*x).end(),
                std::make_pair(kernel_id, block_id)) != (*x).end()) {
    return true;
  } else {
    return false;
  }
}

bool judge_format_compute_block_id(int kernel_id, int block_id,
                                   std::vector<std::pair<int, int>> *x) {
  std::pair<int, int> target = std::make_pair(kernel_id, block_id);
  if (std::find(x->begin(), x->end(), target) != x->end())
    return true;
  else
    return false;
}

void trace_parser::process_mem_instns(const std::string mem_instns_dir,
                                      bool PRINT_LOG,
                                      std::vector<std::pair<int, int>> *x) {
  mem_instns.resize(appcfg.get_kernels_num());
  for (unsigned kid = 0; kid < appcfg.get_kernels_num(); ++kid)
    mem_instns[kid].resize(appcfg.get_kernel_grid_size(kid));

  DIR *dir;
  struct dirent *entry;

  if ((dir = opendir(mem_instns_dir.c_str())) == nullptr)
    std::cerr << "Not exist directory " << mem_instns_dir << ", please check."
              << std::endl;

  static const std::regex pattern(R"(kernel_(\d+)_block_(\d+)\.mem)");
  std::smatch match;

  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG && judge_format_mem(entry->d_name, x) &&
        entry->d_name[strlen(entry->d_name) - 4] == '.' &&
        entry->d_name[strlen(entry->d_name) - 3] == 'm' &&
        entry->d_name[strlen(entry->d_name) - 2] == 'e' &&
        entry->d_name[strlen(entry->d_name) - 1] == 'm') {
      auto search = std::string(entry->d_name);
      std::string mem_instns_filepath = mem_instns_dir + "/" + entry->d_name;

      if (std::regex_search(search, match, pattern)) {
        int kernel_id = std::stoi(match[1]);
        int block_id = std::stoi(match[2]);

        std::ifstream fs;
        fs.open(mem_instns_filepath);
        if (!fs.is_open()) {
          std::cout << "Unable to open file: " << mem_instns_filepath
                    << std::endl;
          exit(1);
        }
        std::string line;
        while (!fs.eof()) {
          getline(fs, line);
          if (line.empty())
            continue;
          else {

            int _addr_groups;
            unsigned _pc, _time_stamp;
            std::string _opcode;
            unsigned _mask;
            unsigned long long _addr_start1, _addr_start2;
            unsigned stride_num_pairs_1, stride_num_pairs_2;

            std::stringstream ss;
            ss.str(line);

            ss >> std::hex >> _pc;
            ss >> _opcode;
            ss >> std::hex >> _mask;
            ss >> std::hex >> _time_stamp;
            ss >> std::hex >> _addr_groups;
            ss >> std::hex >> _addr_start1;
            ss >> std::hex >> stride_num_pairs_1;

            std::vector<long long> _stride_num;
            std::string tmp_stride_num;

            for (unsigned _i = 0; _i < stride_num_pairs_1; _i++) {
              ss >> tmp_stride_num;
              size_t pos = tmp_stride_num.find(':');

              long long stride =
                  std::stoll(tmp_stride_num.substr(0, pos), nullptr, 10);
              int num = std::stoi(tmp_stride_num.substr(pos + 1), nullptr, 10);

              for (int _j = 0; _j < num; _j++) {
                _stride_num.push_back(stride);
              }
            }

            if (_addr_groups == 2) {
              ss >> std::hex >> _addr_start2;
              ss >> std::hex >> stride_num_pairs_2;
              for (unsigned _i = 0; _i < stride_num_pairs_2; _i++) {
                ss >> tmp_stride_num;
                size_t pos = tmp_stride_num.find(':');
                long long stride =
                    std::stoll(tmp_stride_num.substr(0, pos), nullptr, 10);
                int num =
                    std::stoi(tmp_stride_num.substr(pos + 1), nullptr, 10);
                for (int _j = 0; _j < num; _j++) {
                  _stride_num.push_back(stride);
                }
              }
            }

            mem_instns[kernel_id - 1][block_id].push_back(
                mem_instn(_pc, _addr_start1, _time_stamp, _addr_groups,
                          _addr_start2, _mask, _opcode, &_stride_num));
          }
        }
        fs.close();
      } else {
        std::cerr << "Wrong name format of memory trace file: " << entry->d_name
                  << std::endl;
      }
    }
  }

  closedir(dir);
}

void trace_parser::read_mem_instns(bool PRINT_LOG,
                                   std::vector<std::pair<int, int>> *x) {
  if (configs_filepath.back() == '/') {
    mem_instns_dir = configs_filepath + "../memory_traces";
  } else {
    mem_instns_dir = configs_filepath + "/" + "../memory_traces";
  }

  mem_instns.resize(appcfg.get_kernels_num());
  for (unsigned kid = 0; kid < appcfg.get_kernels_num(); ++kid) {
    mem_instns[kid].resize(appcfg.get_kernel_grid_size(kid));
  }

  process_mem_instns(mem_instns_dir, PRINT_LOG, x);
}

void trace_parser::process_compute_instns(std::string compute_instns_dir,
                                          bool PRINT_LOG,
                                          std::vector<std::pair<int, int>> *x) {

  DIR *dir;
  struct dirent *entry;

  if ((dir = opendir(compute_instns_dir.c_str())) == nullptr)
    std::cerr << "Not exist directory " << compute_instns_dir
              << ", please check." << std::endl;

  static const std::regex pattern(R"(kernel_(\d+)\.sass)");
  std::smatch match;

  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG &&
        judge_format_compute_kernel_id(entry->d_name, x)) {
      auto search = std::string(entry->d_name);
      std::string compute_instns_filepath =
          compute_instns_dir + "/" + entry->d_name;

      if (std::regex_search(search, match, pattern)) {
        int kernel_id = std::stoi(match[1]);

        std::ifstream fs;
        fs.open(compute_instns_filepath);
        if (!fs.is_open()) {
          std::cout << "Unable to open file: " << compute_instns_filepath
                    << std::endl;
          exit(1);
        }
        std::string line;
        while (!fs.eof()) {
          getline(fs, line);
          if (line.empty())
            continue;
          else {
            std::string _pc_str;
            unsigned _pc;

            std::string _mask_str;
            unsigned _mask;

            std::string _gwarp_id_str;
            unsigned _gwarp_id;

            std::stringstream ss;
            ss.str(line);

            while (ss >> _pc_str >> _mask_str >> _gwarp_id_str) {
              _gwarp_id = std::stoi(_gwarp_id_str, nullptr, 16);

              if (1) {
                _pc = std::stoi(_pc_str, nullptr, 16);
                if (_pc_str == " " || _pc_str == "" || _pc_str == "\n")
                  break;
                if (_mask_str == "!")
                  _mask = 0xffffffff;
                else {
                  _mask = std::stoi(_mask_str, nullptr, 16);
                }

                _inst_trace_t *_inst_trace =
                    (*get_instncfg()->get_instn_info_vector())[std::make_pair(
                        kernel_id - 1, _pc)];

                conpute_instns[kernel_id - 1][_gwarp_id].push_back(
                    compute_instn(kernel_id - 1, _pc, _mask, _gwarp_id,
                                  _inst_trace));
              }
            }
          }
        }
        fs.close();
      } else {
        std::cerr << "Wrong name format of memory trace file: " << entry->d_name
                  << std::endl;
      }
    }
  }

  closedir(dir);
}

void trace_parser::process_compute_instns_fast(
    std::string compute_instns_dir, bool PRINT_LOG,
    std::vector<std::pair<int, int>> *x) {

  DIR *dir;
  struct dirent *entry;

  if ((dir = opendir(compute_instns_dir.c_str())) == nullptr)
    std::cerr << "Not exist directory " << compute_instns_dir
              << ", please check." << std::endl;

  static const std::regex pattern(
      R"(kernel_(\d+)_gwarp_id_(\d+)\.split\.sass)");
  std::smatch match;

  while ((entry = readdir(dir)) != nullptr) {

    if (entry->d_type == DT_REG &&
        entry->d_name[strlen(entry->d_name) - 6] == 't') {

      auto search = std::string(entry->d_name);

      if (std::regex_search(search, match, pattern)) {
        int kernel_id = std::stoi(match[1]);

        int gwarp_id = std::stoi(match[2]);

        int num_warps_per_block =
            get_appcfg()->get_num_warp_per_block(kernel_id - 1);

        int block_id = (int)(gwarp_id / num_warps_per_block);

        if (!judge_format_compute_kernel_id_fast(kernel_id, block_id, x))
          continue;

        std::string compute_instns_filepath =
            compute_instns_dir + "/" + entry->d_name;

        std::ifstream fs(compute_instns_filepath);

        if (!fs.is_open()) {
          std::cout << "Unable to open file: " << compute_instns_filepath
                    << std::endl;
          exit(1);
        }

        char buf[BUFSIZ * 10];
        fs.rdbuf()->pubsetbuf(buf, sizeof(buf));

        std::string line;
        while (!fs.eof()) {
          getline(fs, line);
          if (line.empty())
            continue;
          else {
            unsigned _pc;

            std::string _mask_str;
            unsigned _mask;

            char mask_str[9];

            std::istringstream iss(line);
            iss >> std::hex >> _pc >> mask_str;

            _mask_str = std::string(mask_str);

            if (_mask_str == "!")
              _mask = 0xffffffff;
            else {
              _mask = (unsigned)std::stoul(_mask_str, nullptr, 16);
            }

            _inst_trace_t *_inst_trace =
                (*get_instncfg()->get_instn_info_vector())[std::make_pair(
                    kernel_id - 1, _pc)];

            conpute_instns[kernel_id - 1][gwarp_id].emplace_back(compute_instn(
                kernel_id - 1, _pc, _mask, gwarp_id, _inst_trace, NULL));
          }
        }
        fs.close();

      } else {
        std::cerr << "Wrong name format of memory trace file: " << entry->d_name
                  << std::endl;
      }
    }
  }

  closedir(dir);
}

void trace_parser::read_compute_instns(bool PRINT_LOG,
                                       std::vector<std::pair<int, int>> *x) {
  if (configs_filepath.back() == '/') {
    compute_instns_dir = configs_filepath + "../sass_traces";
  } else {
    compute_instns_dir = configs_filepath + "/" + "../sass_traces";
  }

  conpute_instns.resize(appcfg.get_kernels_num());
  for (unsigned kid = 0; kid < appcfg.get_kernels_num(); ++kid)
    conpute_instns[kid].resize(appcfg.get_num_global_warps(kid));

  process_compute_instns_fast(compute_instns_dir, PRINT_LOG, x);
}

void split(const std::string &str, std::vector<std::string> &cont,
           char delimi = ' ') {
  std::stringstream ss(str);
  std::string token;
  while (std::getline(ss, token, delimi)) {
    cont.push_back(token);
  }
}

void trace_parser::parse_memcpy_info(const std::string &memcpy_command,
                                     size_t &address, size_t &count) {
  std::vector<std::string> params;
  split(memcpy_command, params, ',');
  assert(params.size() == 3);
  std::stringstream ss;
  ss.str(params[1]);
  ss >> std::hex >> address;
  ss.clear();
  ss.str(params[2]);
  ss >> std::dec >> count;
}

std::vector<std::vector<inst_trace_t> *>
trace_parser::get_next_threadblock_traces(unsigned trace_version,
                                          unsigned enable_lineinfo,
                                          std::ifstream *ifs,
                                          const std::string kernel_name,
                                          unsigned kernel_id,
                                          unsigned num_warps_per_thread_block) {
  std::vector<std::vector<inst_trace_t> *> threadblock_traces;
  threadblock_traces.resize(num_warps_per_thread_block);
  unsigned block_id_x = 0, block_id_y = 0, block_id_z = 0;
  bool start_of_tb_stream_found = false;

  unsigned warp_id = 0;
  unsigned insts_num = 0;
  unsigned inst_count = 0;

  while (!ifs->eof()) {
    std::string line;
    std::stringstream ss;
    std::string string1, string2;

    getline(*ifs, line);

    if (line.length() == 0) {
      continue;
    } else {
      ss.str(line);
      ss >> string1 >> string2;
      if (string1 == "#BEGIN_TB") {
        if (!start_of_tb_stream_found) {
          start_of_tb_stream_found = true;
        } else
          assert(0 &&
                 "Parsing error: thread block start before the previous one "
                 "finishes");
      } else if (string1 == "#END_TB") {
        assert(start_of_tb_stream_found);
        break;
      } else if (string1 == "thread" && string2 == "block") {
        assert(start_of_tb_stream_found);
        sscanf(line.c_str(), "thread block = %u,%u,%u", &block_id_x,
               &block_id_y, &block_id_z);
        std::cout << "Parsing trace of " << line << "..." << std::endl;
      } else if (string1 == "warp") {

        assert(start_of_tb_stream_found);
        sscanf(line.c_str(), "warp = %u", &warp_id);
      } else if (string1 == "insts") {
        assert(start_of_tb_stream_found);
        sscanf(line.c_str(), "insts = %u", &insts_num);
        threadblock_traces[warp_id] = new std::vector<inst_trace_t>();
        threadblock_traces[warp_id]->resize(insts_num);
        inst_count = 0;
      } else {
        assert(start_of_tb_stream_found);
        threadblock_traces[warp_id]
            ->at(inst_count)
            .parse_from_string(line, trace_version, enable_lineinfo,
                               kernel_name, kernel_id);
        inst_count++;
      }
    }
  }
  return threadblock_traces;
}
