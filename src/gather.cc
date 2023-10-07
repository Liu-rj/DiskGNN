#include <fcntl.h>
#include <omp.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

#include "gather.h"

namespace offgs {
torch::Tensor GatherMemMap(const torch::Tensor& features,
                           const torch::Tensor& idx, int64_t feature_dim) {
  // open file
  int64_t feature_size = feature_dim * sizeof(float);

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  torch::Tensor ret = torch::empty({idx.numel(), feature_dim}, options);

  auto features_data = features.data_ptr<float>();
  auto idx_data = idx.data_ptr<int64_t>();
  auto ret_data = ret.data_ptr<float>();

#pragma omp parallel for num_threads(64)
  for (int64_t i = 0; i < idx.numel(); i++) {
    memcpy(ret_data + feature_dim * i,
           features_data + idx_data[i] * feature_dim, feature_size);
  }

  return ret;
}

torch::Tensor GatherPRead(const std::string& feature_file,
                          const torch::Tensor& idx, int64_t feature_dim) {
  int feature_fd = open(feature_file.c_str(), O_RDONLY);
  int64_t feature_size = feature_dim * sizeof(float);

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  torch::Tensor ret = torch::empty({idx.numel(), feature_dim}, options);

  auto idx_data = idx.data_ptr<int64_t>();
  auto ret_data = ret.data_ptr<float>();

#pragma omp parallel for num_threads(64)
  for (int64_t i = 0; i < idx.numel(); i++) {
    if (pread(feature_fd, ret_data + feature_dim * i, feature_size,
              idx_data[i] * feature_dim) == -1) {
      fprintf(stderr, "ERROR: %s\n", strerror(errno));
    }
  }

  return ret;
}
}  // namespace offgs