#ifndef OFFGS_LOAD_H_
#define OFFGS_LOAD_H_

#include <torch/torch.h>
#include <vector>

namespace offgs {
std::vector<torch::Tensor> LoadFeats(const std::string& file_path,
                                     int64_t feature_dim, int64_t num_align,
                                     int64_t omp_threads);

std::vector<torch::Tensor> LoadFeats_Direct(const std::string& file_path,
                                            int64_t num_indices,
                                            int64_t feature_dim);

torch::Tensor LoadTensor(const std::string& file_path);
}  // namespace offgs

#endif