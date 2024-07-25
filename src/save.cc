#include <c10/util/Logging.h>
#include <fcntl.h>
#include <omp.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <fstream>
#include <thread>

#include "save.h"
#include "utils.h"

namespace offgs {
template <typename DType>
void SaveFeatsImpl(const std::string& file_path, const torch::Tensor& feature) {
  int fd =
      open(file_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    LOG(FATAL) << "failed to open or create file";
    return;
  }

  auto buf = feature.data_ptr<DType>();
  auto bytes_left = feature.numel() * sizeof(DType);

  while (bytes_left > 0) {
    auto trans = write(fd, buf, bytes_left);
    if (trans == -1) {
      LOG(FATAL) << "ERROR: " << strerror(errno);
    }
    buf += trans / sizeof(DType);
    bytes_left -= trans;
  }

  close(fd);
}

void SaveFeats(const std::string& file_path, const torch::Tensor& feature) {
  auto scalar_type = feature.scalar_type();
  FLOAT_TYPE_SWITCH(scalar_type, DType,
                    { SaveFeatsImpl<DType>(file_path, feature); });
}

template <typename DType>
void SaveFeatsAppendImpl(const std::string& file_path,
                         const torch::Tensor& feature) {
  int fd =
      open(file_path.c_str(), O_CREAT | O_WRONLY | O_APPEND, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    LOG(FATAL) << "failed to open or create file";
    return;
  }

  auto buf = feature.data_ptr<DType>();
  auto bytes_left = feature.numel() * sizeof(DType);

  while (bytes_left > 0) {
    auto trans = write(fd, buf, bytes_left);
    if (trans == -1) {
      LOG(FATAL) << "ERROR: " << strerror(errno);
    }
    buf += trans / sizeof(DType);
    bytes_left -= trans;
  }

  close(fd);
}

void SaveFeatsAppend(const std::string& file_path,
                     const torch::Tensor& feature) {
  auto scalar_type = feature.scalar_type();
  FLOAT_TYPE_SWITCH(scalar_type, DType,
                    { SaveFeatsAppendImpl<DType>(file_path, feature); });
}
}  // namespace offgs