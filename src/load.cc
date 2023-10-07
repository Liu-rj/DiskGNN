#include <c10/util/Logging.h>
#include <fcntl.h>
#include <omp.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <fstream>
#include <thread>

#include "load.h"

#define ALIGNMENT 4096

namespace offgs {
std::vector<torch::Tensor> LoadFeats(const std::string& file_path,
                                     int64_t feature_dim, int64_t num_align,
                                     int64_t omp_theads) {
  std::ifstream is(file_path, std::ifstream::binary | std::ifstream::ate);
  if (!is.is_open()) {
    LOG(FATAL) << "failed to open file";
    return {};
  }
  std::size_t data_size = is.tellg();
  is.close();

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto all_data = torch::empty(data_size / sizeof(float), options);
  auto data_ptr = all_data.data_ptr<float>();

  auto read_unit = num_align * ALIGNMENT;
  auto num_reads = static_cast<int>((data_size + read_unit - 1) / read_unit);
  // const auto processor_count = std::thread::hardware_concurrency();

  int fd = open(file_path.c_str(), O_RDONLY);

#pragma omp parallel for num_threads(omp_theads)
  for (int i = 0; i < num_reads; i++) {
    if (pread(fd, data_ptr + (read_unit * i) / sizeof(float), read_unit,
              read_unit * i) == -1) {
      LOG(FATAL) << "ERROR: " << strerror(errno);
    }
  }

  close(fd);

  auto sizes = all_data.slice(0, 0, 5).to(torch::kInt32).cumsum(0);
  auto sizes_ptr = sizes.data_ptr<int64_t>();

  std::vector<torch::Tensor> res;
  res.push_back(all_data.slice(0, 5, sizes_ptr[0] + 5)
                    .clone()
                    .view({sizes_ptr[0] / feature_dim, feature_dim}));
  for (int i = 0; i < 4; i++) {
    res.push_back(all_data.slice(0, sizes_ptr[i] + 5, sizes_ptr[i + 1] + 5)
                      .clone()
                      .to(torch::kInt32));
  }

  return res;
}

std::vector<torch::Tensor> LoadFeats_ODirect(const std::string& file_path,
                                             int64_t feature_dim,
                                             int64_t num_align,
                                             int64_t omp_theads) {
  std::ifstream is(file_path, std::ifstream::binary | std::ifstream::ate);
  if (!is.is_open()) {
    LOG(FATAL) << "failed to open file";
    return {};
  }
  std::size_t data_size = is.tellg();
  is.close();

  auto read_unit = num_align * ALIGNMENT;
  auto reminder = data_size % read_unit;
  auto aligned_size =
      reminder == 0 ? data_size : data_size - reminder + read_unit;
  float* read_buffer = (float*)aligned_alloc(ALIGNMENT, aligned_size);
  auto num_reads = static_cast<int>(aligned_size / read_unit);
  // const auto processor_count = std::thread::hardware_concurrency();

  int fd = open(file_path.c_str(), O_RDONLY | O_DIRECT);

#pragma omp parallel for num_threads(omp_theads)
  for (int i = 0; i < num_reads; i++) {
    if (pread(fd, read_buffer + (read_unit * i) / sizeof(float), read_unit,
              read_unit * i) == -1) {
      LOG(FATAL) << "ERROR: " << strerror(errno);
    }
  }

  close(fd);

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto all_data =
      torch::from_blob(read_buffer, data_size / sizeof(float), options);
  auto sizes = all_data.slice(0, 0, 5).to(torch::kInt32).cumsum(0);
  auto sizes_ptr = sizes.data_ptr<int64_t>();

  std::vector<torch::Tensor> res;
  res.push_back(all_data.slice(0, 5, sizes_ptr[0] + 5)
                    .clone()
                    .view({sizes_ptr[0] / feature_dim, feature_dim}));
  for (int i = 0; i < 4; i++) {
    res.push_back(all_data.slice(0, sizes_ptr[i] + 5, sizes_ptr[i + 1] + 5)
                      .clone()
                      .to(torch::kInt32));
  }

  free(read_buffer);

  return res;
}

torch::Tensor LoadTensor(const std::string& file_path) {
  std::ifstream is(file_path, std::ifstream::binary);

  if (!is.is_open()) {
    LOG(FATAL) << "failed to open file";
    return {};
  }

  is.seekg(0, std::ios_base::end);
  std::size_t size = is.tellg();
  is.seekg(0, std::ios_base::beg);

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  torch::Tensor ret = torch::empty(size / sizeof(float), options);
  auto data_ptr = ret.data_ptr<float>();

  if (!is.read((char*)data_ptr, size)) {
    LOG(FATAL) << "failed to read from file";
    return {};
  }
  is.close();

  return ret;
}
}  // namespace offgs