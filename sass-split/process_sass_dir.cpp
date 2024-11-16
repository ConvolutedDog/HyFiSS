#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <tuple>
#include <sstream>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include <string>
#include <regex>

/// Due to the limitations of the operating system, enabling too many
/// parallel writes of files at the same time will cause some files
/// to fail to be written, so we set the maximum number of files that
/// can be written at the same time is `512`.
#define MAX_SIZE_OF_MME_TRACE_FP_MAP 512

std::vector<std::string> split(const std::string &str) {
  std::istringstream iss(str);
  std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},
                                  std::istream_iterator<std::string>{}};
  return tokens;
}

/// Stitch together `path` and `file` to form an absolute path.
inline std::string getFullPath(const std::string &path,
                               const std::string &file) {
  return path + "/" + file;
}

/// This function is used to check if a file exists at a given path.
/// It obtains the information of the file through the `stat` system
/// call, and if the `stat` function returns 0, indicating that the
/// file exists, this function returns true; If the file does not
/// exist or an error occurs, stat returns a non-0 value and this
/// function returns false.
bool fileExists(const std::string &path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

/// To determine whether the current `file` is a file that needs to
/// be processed, because the goal of this application is to divide
/// 'kernel_x.sass' into thread blocks.
bool isSassFile(const std::string &file) {
  // The pending source file name is like 'kernel_x.sass', and the
  // processed target file name is like 'kernel_x_block_y.sass'.
  const std::string sass_ext = ".sass";
  // If the file name has '_block_' in it, it is not the source file
  // we are working on.
  const std::string split_sass_ext = "_block_";

  if (file.size() >= sass_ext.size() &&
      file.substr(file.size() - sass_ext.size()) == sass_ext) {
    if (file.find(split_sass_ext) != std::string::npos) {
      return false;
    }
    return true;
  }
  return false;
}

/// To determine whether the current `file` is a file that needs to
/// be processed, because the goal of this application is to divide
/// 'kernel_x.mem' into thread blocks.
bool isMemoryFile(const std::string &file) {
  // The pending source file name is like 'kernel_x.mem', and the
  // processed target file name is like 'kernel_x_block_y.mem'.
  const std::string mem_ext = ".mem";
  // If the file name has '_block_' in it, it is not the source file
  // we are working on.
  const std::string split_mem_ext = "_block_";

  if (file.size() >= mem_ext.size() &&
      file.substr(file.size() - mem_ext.size()) == mem_ext) {
    if (file.find(split_mem_ext) != std::string::npos) {
      return false;
    }
    return true;
  }
  return false;
}

/// Due to the limitations of the operating system, enabling too many
/// parallel writes of files at the same time will cause some files
/// to fail to be written, so we set the maximum number of files that
/// can be written at the same time is `512`. We set up a dictionary
/// for the open handle for up to `512` files, and its key is tuple
/// {kernel index, block index}.
std::map<std::tuple<int, int>, std::ofstream> sass_trace_fp_map;
/// Refer to `sass_trace_fp_map`.
std::map<std::tuple<int, int>, std::ofstream> mem_trace_fp_map;

/// TODO: Change the target outputPath to be like 'kernel_x_block_y.sass'.
/// May need to read 'sass_dir/../configs/app.config' and get the value of
/// `-kernel_x_block_size`, calculate the block index of each global warp
/// as `kernel_x_block_size/32`.
std::map<int, int> kernel_block_size_map;

void setKernelBlockSizeMap(const std::string &sass_dir) {
  std::string appConfigFile = sass_dir + "/../configs/app.config";
  std::ifstream file(appConfigFile);
  if (!file.is_open()) {
    std::cerr << "Cannot open app.config file: " << appConfigFile << std::endl;
    abort();
  }

  std::vector<std::string> lines;
  std::string line;
  while (getline(file, line)) {
    lines.push_back(line);
  }

  for (unsigned kernel_id = 0; kernel_id < 100; ++kernel_id) {
    std::string entry = std::string("-kernel_") + 
                        std::to_string(kernel_id + 1) + 
                        std::string("_block_size (\\d+)");
    std::regex pattern(entry);
    std::smatch match;
    std::string line;

    for (auto &line : lines) {
      if (std::regex_search(line, match, pattern)) {
        if (match.size() > 1) {
          std::string num_str = match.str(1);
          int num = std::stoi(num_str);
          kernel_block_size_map[kernel_id + 1] = num;
        }
      }
    }
  }
}

bool is_directory_exists(const std::string &dir_path) {
  struct stat statbuf;
  if (stat(dir_path.c_str(), &statbuf) != -1) {
    if (S_ISDIR(statbuf.st_mode)) return true;
  }
  return false;
}

void remove_directory_contents(const std::string &dir_path) {
  DIR *dir = opendir(dir_path.c_str());
  struct dirent *next_file;
  char filepath[1024];

  while ((next_file = readdir(dir)) != NULL) {
    if (strcmp(next_file->d_name, ".") == 0 || strcmp(next_file->d_name, "..") == 0) {
      continue;
    }
    snprintf(filepath, sizeof(filepath), "%s/%s", dir_path.c_str(), next_file->d_name);
    remove(filepath);
  }
  closedir(dir);
}

bool user_confirms() {
  std::string input;
  while (true) {
    std::cout << "The target traces already exists, whether "
                 "to overwrite them (y or n): ";
    std::cin >> input;
    if (input == "y") {
      return true;
    } else if (input == "n") {
      exit(0);
    } else {
      std::cout << "Invalid input, please try again.\n";
    }
  }
}

bool first_user_confirms = false;
void create_directory_if_not_exists(const std::string &dir_path) {
  if (!is_directory_exists(dir_path)) {
    mkdir(dir_path.c_str(), 0777);
  } else {
    if (!first_user_confirms) {
      first_user_confirms = true;
      user_confirms();
    }
    remove_directory_contents(dir_path);
  }
}

int main(int argc, char *argv[]) {
  // Expects exactly two arguments: "--dir" followed by the directory
  // path containing sass files.
  if (argc != 3 || std::string(argv[1]) != "--dir") {
    std::cerr << "Usage: " << argv[0] << " --dir <directory_of_sass_files>\n";
    return 1;
  }

  std::string sass_dir = argv[2];

  setKernelBlockSizeMap(sass_dir);

  DIR *dir;
  struct dirent *ent;
  // Absolute paths to the source file to be processed.
  std::vector<std::string> sass_files;

  if ((dir = opendir(sass_dir.c_str())) != NULL) {
    while ((ent = readdir(dir)) != NULL) {
      if (isSassFile(ent->d_name)) {
        std::cout << "Found sass file: " << ent->d_name << std::endl;
        sass_files.push_back(getFullPath(sass_dir, std::string(ent->d_name)));
      } else {
        if (strcmp(ent->d_name, ".") != 0 && strcmp(ent->d_name, "..") != 0) {
          std::string filePath =
              getFullPath(sass_dir, std::string(ent->d_name));

          if (std::remove(filePath.c_str()) == 0) {
            // std::cout << "Deleted non-sass file: " << ent->d_name << std::endl;
          } else {
            // std::cerr << "Error deleting file: " << ent->d_name << std::endl;
          }
        }
      }
    }
    closedir(dir);
  } else {
    perror("");
    return EXIT_FAILURE;
  }

  // 2024.04.07 Start

  std::string memory_dir = argv[2] + std::string("/../memory_traces");;
  // Absolute paths to the source file to be processed.
  std::vector<std::string> memory_files;
  if ((dir = opendir(memory_dir.c_str())) != NULL) {
    while ((ent = readdir(dir)) != NULL) {
      if (isMemoryFile(ent->d_name)) {
        std::cout << "Found mem file: " << ent->d_name << std::endl;
        memory_files.push_back(getFullPath(memory_dir, std::string(ent->d_name)));
      } else {
        if (strcmp(ent->d_name, ".") != 0 && strcmp(ent->d_name, "..") != 0) {
          std::string filePath =
              getFullPath(memory_dir, std::string(ent->d_name));

          if (std::remove(filePath.c_str()) == 0) {
            // std::cout << "Deleted non-mem file: " << ent->d_name << std::endl;
          } else {
            // std::cerr << "Error deleting file: " << ent->d_name << std::endl;
          }
        }
      }
    }
    closedir(dir);
  } else {
    perror("");
    return EXIT_FAILURE;
  }

  // 2024.04.07 End

  for (const auto &sass_file : sass_files) {
    std::map<std::pair<int, int>, std::vector<std::string>> warp_content;
    std::cout << "Processing " << sass_file << "\n";
    std::ifstream file(sass_file);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    auto tokens = split(content);
    auto underscorePos = sass_file.find_last_of('_');
    int kernel_id = std::stoi(sass_file.substr(
        underscorePos + 1, sass_file.find(".sass") - underscorePos - 1));
    if (kernel_id > 100) continue;

    for (size_t i = 0; i < tokens.size() / 3; ++i) {
      int gwarp_id = std::stoi(tokens[i * 3 + 2], nullptr, 16);

      if (warp_content.find({kernel_id, gwarp_id}) == warp_content.end()) {
        warp_content[{kernel_id, gwarp_id}] = std::vector<std::string>();
      }

      warp_content[{kernel_id, gwarp_id}].push_back(tokens[i * 3] + " " + // pc
                                                    tokens[i * 3 + 1] + " " + // mask
                                                    std::to_string(gwarp_id) + "\n"); // gwarp_id
    }

    file.close();

    std::string outputParentPath = sass_dir + "/kernel-" + std::to_string(kernel_id);
    create_directory_if_not_exists(outputParentPath);

    for (auto &item : warp_content) {
      unsigned block_id = (unsigned)(
        item.first.second * 32 / kernel_block_size_map[item.first.first]);

      std::string outputPath =
        outputParentPath + "/kernel_" + std::to_string(kernel_id) + "_block_" +
        std::to_string(block_id) + ".sass";

      std::ofstream *sass_trace_fp_ptr = nullptr;

      auto x = std::make_tuple(kernel_id, block_id);
      auto it_map = sass_trace_fp_map.find(x);
      if (it_map == sass_trace_fp_map.end()) {
        if (sass_trace_fp_map.size() >= MAX_SIZE_OF_MME_TRACE_FP_MAP) {
          auto it = sass_trace_fp_map.begin();
          it->second.close();
          sass_trace_fp_map.erase(it);
        }
        std::ofstream &sass_trace_fp =
            sass_trace_fp_map.emplace(x, std::ofstream{}).first->second;
        sass_trace_fp.open(outputPath, std::ios::app);
        sass_trace_fp_ptr = &sass_trace_fp;
      } else {
        sass_trace_fp_ptr = &(it_map->second);
      }

      auto &f_open = *sass_trace_fp_ptr;

      for (auto &line : item.second) {
        f_open << line;
      }
    }
    auto it_map = sass_trace_fp_map.begin();
    while (it_map != sass_trace_fp_map.end()) {
      it_map->second.close();
      it_map = sass_trace_fp_map.erase(it_map);
    }
  }

  // 2024.04.07 Start

  for (const auto &mem_file : memory_files) {
    std::map<std::pair<int, int>, std::vector<std::string>> blk_content;
    std::cout << "Processing " << mem_file << "\n";
    std::ifstream file(mem_file);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    
    std::istringstream iss(content);
    
    auto underscorePos = mem_file.find_last_of('_');
    int kernel_id = std::stoi(mem_file.substr(
        underscorePos + 1, mem_file.find(".mem") - underscorePos - 1));
    if (kernel_id > 100) continue;

    std::string line_str;
    int block_id;

    while (std::getline(iss, line_str)) {
      std::istringstream line_iss(line_str);
      std::string block_id_str;
      
      if (std::getline(line_iss, block_id_str, ' ')) {
        block_id = std::stoi(block_id_str, nullptr, 16);
        std::string remaining_content((std::istreambuf_iterator<char>(line_iss)),
                                       std::istreambuf_iterator<char>());
        if (blk_content.find({kernel_id, block_id}) == blk_content.end()) {
          blk_content[{kernel_id, block_id}] = std::vector<std::string>();
        }
        blk_content[{kernel_id, block_id}].push_back(remaining_content);
      }
    }

    file.close();

    std::string outputParentPath = memory_dir + "/kernel-" + std::to_string(kernel_id);
    create_directory_if_not_exists(outputParentPath);

    for (auto &item : blk_content) {
      std::string outputPath =
          outputParentPath + "/kernel_" + std::to_string(item.first.first) +
          "_block_" + std::to_string(item.first.second) + ".mem";

      std::ofstream *mem_trace_fp_ptr = nullptr;

      auto x = std::make_tuple(kernel_id, item.first.second);
      auto it_map = mem_trace_fp_map.find(x);
      if (it_map == mem_trace_fp_map.end()) {
        if (mem_trace_fp_map.size() >= MAX_SIZE_OF_MME_TRACE_FP_MAP) {
          auto it = mem_trace_fp_map.begin();
          it->second.close();
          mem_trace_fp_map.erase(it);
        }
        std::ofstream &mem_trace_fp =
            mem_trace_fp_map.emplace(x, std::ofstream{}).first->second;
        mem_trace_fp.open(outputPath, std::ios::app);
        mem_trace_fp_ptr = &mem_trace_fp;
      } else {
        mem_trace_fp_ptr = &(it_map->second);
      }

      auto &f_open = *mem_trace_fp_ptr;

      for (auto &line : item.second) {
        f_open << line << std::endl;
      }
    }
    auto it_map = mem_trace_fp_map.begin();
    while (it_map != mem_trace_fp_map.end()) {
        it_map->second.close();
        it_map = mem_trace_fp_map.erase(it_map);
    }
  }

  // 2024.04.07 End

  return 0;
}
