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

#define MAX_SIZE_OF_MME_TRACE_FP_MAP 512

std::vector<std::string> split(const std::string &str) {
  std::istringstream iss(str);
  std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},
                                  std::istream_iterator<std::string>{}};
  return tokens;
}

std::string getFullPath(const std::string &path, const std::string &file) {
  return path + "/" + file;
}

bool fileExists(const std::string &path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

bool isSassFile(const std::string &file) {
  const std::string sass_ext = ".sass";
  const std::string split_sass_ext = ".split.sass";

  if (file.size() >= sass_ext.size() &&
      file.substr(file.size() - sass_ext.size()) == sass_ext) {

    if (file.size() >= split_sass_ext.size() &&
        file.substr(file.size() - split_sass_ext.size()) == split_sass_ext) {
      return false;
    }
    return true;
  }
  return false;
}

// 2024.04.07 Start
bool isMemoryFile(const std::string &file) {
  const std::string mem_ext = ".mem";
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
// 2024.04.07 End

std::map<std::tuple<int, int>, std::ofstream> sass_trace_fp_map;
std::map<std::tuple<int, int>, std::ofstream> mem_trace_fp_map;

int main(int argc, char *argv[]) {
  if (argc != 3 || std::string(argv[1]) != "--dir") {
    std::cerr << "Usage: " << argv[0] << " --dir <directory_of_sass_files>\n";
    return 1;
  }

  std::string sass_dir = argv[2];
  DIR *dir;
  struct dirent *ent;
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
  std::string memory_dir = argv[2] + std::string("/../memory_traces");
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

      warp_content[{kernel_id, gwarp_id}].push_back(tokens[i * 3] + " " +
                                                    tokens[i * 3 + 1] + "\n");
    }

    file.close();

    for (auto &item : warp_content) {
      std::string outputPath =
          sass_dir + "/kernel_" + std::to_string(item.first.first) +
          "_gwarp_id_" + std::to_string(item.first.second) + ".split.sass";
      
      std::ofstream *sass_trace_fp_ptr = nullptr;

      auto x = std::make_tuple(kernel_id, item.first.second);
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

    for (auto &item : blk_content) {
      std::string outputPath =
          memory_dir + "/kernel_" + std::to_string(item.first.first) +
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
