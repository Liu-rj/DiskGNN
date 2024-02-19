#ifndef OFFGS_SAVE_H_
#define OFFGS_SAVE_H_

#include <torch/torch.h>
#include <vector>

namespace offgs {
void SaveFeats(const std::string& file_path, const torch::Tensor& feature);
}  // namespace offgs

#endif  // OFFGS_SAVE_H