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
                                     int64_t omp_threads) {
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

  auto block_size = num_align * ALIGNMENT;
  auto num_reads = static_cast<int>((data_size + block_size - 1) / block_size);
  // const auto processor_count = std::thread::hardware_concurrency();

  int fd = open(file_path.c_str(), O_RDONLY);

#pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < num_reads; i++) {
    if (pread(fd, data_ptr + (block_size * i) / sizeof(float), block_size,
              block_size * i) == -1) {
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

std::vector<torch::Tensor> LoadFeats_Direct_OMP(const std::string& file_path,
                                                int64_t feature_dim,
                                                int64_t num_align) {
  std::ifstream is(file_path, std::ifstream::binary | std::ifstream::ate);
  if (!is.is_open()) {
    LOG(FATAL) << "failed to open file";
    return {};
  }
  std::size_t data_size = is.tellg();
  is.close();

  auto block_size = num_align * ALIGNMENT;
  auto reminder = data_size % ALIGNMENT;
  auto aligned_size =
      reminder == 0 ? data_size : data_size - reminder + ALIGNMENT;
  float* read_buffer = (float*)aligned_alloc(ALIGNMENT, aligned_size);
  auto num_blocks =
      static_cast<int>((aligned_size + block_size - 1) / block_size);
  // const auto processor_count = std::thread::hardware_concurrency();
  auto omp_threads = std::min(num_blocks, 64);

  int fd = open(file_path.c_str(), O_RDONLY | O_DIRECT);

#pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < num_blocks; i++) {
    auto read_size =
        i == (num_blocks - 1) ? aligned_size - block_size * i : block_size;
    if (pread(fd, read_buffer + (block_size * i) / sizeof(float), read_size,
              block_size * i) == -1) {
      LOG(FATAL) << "ERROR: " << strerror(errno);
    }
  }

  close(fd);

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto all_data =
      torch::from_blob(read_buffer, data_size / sizeof(float), options);
  auto sizes = all_data.slice(0, 0, 5).to(torch::kInt64);
  // avoid overflow of size[0]
  sizes[0] = sizes[0] * sizes[1];
  sizes = sizes.cumsum(0);
  auto sizes_ptr = sizes.data_ptr<int64_t>();
  // print all of sizes
  // for (int i = 0; i < 5; i++) {
  //   printf("%d\n", sizes_ptr[i]);
  // }

  std::vector<torch::Tensor> res;
  res.push_back(all_data.slice(0, 5, sizes_ptr[0] + 5)
                    .clone()
                    .view({sizes_ptr[0] / feature_dim, feature_dim}));
  for (int i = 0; i < 4; i++) {
    res.push_back(all_data.slice(0, sizes_ptr[i] + 5, sizes_ptr[i + 1] + 5)
                      .clone()
                      .to(torch::kInt64));
  }

  free(read_buffer);

  return res;
}

torch::Tensor LoadFeats_Direct(const std::string& file_path,
                               int64_t num_indices, int64_t feature_dim) {
  auto total_size = num_indices * feature_dim * sizeof(float);

  size_t reminder = total_size % ALIGNMENT;
  size_t aligned_size =
      reminder == 0 ? total_size : total_size - reminder + ALIGNMENT;
  float* read_buffer = (float*)aligned_alloc(ALIGNMENT, aligned_size);
  size_t residual = aligned_size - total_size;

  int fd = open(file_path.c_str(), O_RDONLY | O_DIRECT);

  auto buf = read_buffer;
  auto bytes_left = aligned_size;
  int64_t offset = 0;
  while (bytes_left > residual) {
    auto trans = pread(fd, buf, bytes_left, offset);
    if (trans == -1) {
      LOG(FATAL) << "ERROR: " << strerror(errno);
    }
    buf += trans / sizeof(float);
    bytes_left -= trans;
    offset += trans;
  }

  close(fd);

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto all_data =
      torch::from_blob(read_buffer, total_size / sizeof(float), options)
          .view({num_indices, feature_dim});
  // all_data = all_data.pin_memory().view({num_indices, feature_dim});

  // free(read_buffer);

  return all_data;
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