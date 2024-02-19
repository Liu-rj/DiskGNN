#include <c10/util/Logging.h>
#include <fcntl.h>
#include <omp.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
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
  while (bytes_left > residual) {
    auto trans = read(fd, buf, bytes_left);
    if (trans == -1) {
      LOG(FATAL) << "ERROR: " << strerror(errno);
    }
    buf += trans / sizeof(float);
    bytes_left -= trans;
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

std::vector<double> LoadDiskCache_Direct_OMP(const std::string& file_path,
                                             const torch::Tensor& out_data,
                                             const torch::Tensor& in_idx,
                                             const torch::Tensor& out_idx,
                                             int64_t feature_dim) {
  struct timespec start, end;

  assert(in_idx.numel() == out_idx.numel());
  auto num_feat_per_page =
      static_cast<int64_t>((ALIGNMENT / sizeof(float)) / feature_dim);

  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  auto ret_tensors = torch::_unique2(
      (in_idx / num_feat_per_page).to(torch::kInt64), true, true);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  auto unique_time = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1000000000.0;

  auto page_ids = std::get<0>(ret_tensors),
       inverse_idx = std::get<1>(ret_tensors);

  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  float* read_buffer =
      (float*)aligned_alloc(ALIGNMENT, page_ids.numel() * ALIGNMENT);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  auto alloc_time = (end.tv_sec - start.tv_sec) +
                    (end.tv_nsec - start.tv_nsec) / 1000000000.0;

  // const auto processor_count = std::thread::hardware_concurrency();
  auto omp_threads = std::min(in_idx.numel(), static_cast<int64_t>(64));

  auto page_ids_ptr = page_ids.data_ptr<int64_t>(),
       inverse_idx_ptr = inverse_idx.data_ptr<int64_t>(),
       in_idx_ptr = in_idx.data_ptr<int64_t>(),
       out_idx_ptr = out_idx.data_ptr<int64_t>();
  auto out_data_ptr = out_data.data_ptr<float>();

  int fd = open(file_path.c_str(), O_RDONLY | O_DIRECT);

  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
#pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < page_ids.numel(); i++) {
    if (pread(fd, read_buffer + (i * ALIGNMENT) / sizeof(float), ALIGNMENT,
              page_ids_ptr[i] * ALIGNMENT) == -1) {
      LOG(FATAL) << "ERROR: " << strerror(errno);
    }
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  auto load_time = (end.tv_sec - start.tv_sec) +
                   (end.tv_nsec - start.tv_nsec) / 1000000000.0;

  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
#pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < in_idx.numel(); i++) {
    auto in_offset = (inverse_idx_ptr[i] * ALIGNMENT) / sizeof(float);
    auto residual = (in_idx_ptr[i] % num_feat_per_page) * feature_dim;
    auto out_offset = out_idx_ptr[i] * feature_dim;
    memcpy(out_data_ptr + out_offset, read_buffer + in_offset + residual,
           feature_dim * sizeof(float));
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  auto copy_time = (end.tv_sec - start.tv_sec) +
                   (end.tv_nsec - start.tv_nsec) / 1000000000.0;

  close(fd);

  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  free(read_buffer);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  auto free_time = (end.tv_sec - start.tv_sec) +
                   (end.tv_nsec - start.tv_nsec) / 1000000000.0;

  return {load_time, copy_time, unique_time, alloc_time, free_time};
}
}  // namespace offgs