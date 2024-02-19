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

namespace offgs {
void SaveFeats(const std::string& file_path, const torch::Tensor& feature) {
  int fd =
      open(file_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    LOG(FATAL) << "failed to open or create file";
    return;
  }

  auto buf = feature.data_ptr<float>();
  auto bytes_left = feature.numel() * sizeof(float);

  while (bytes_left > 0) {
    auto trans = write(fd, buf, bytes_left);
    if (trans == -1) {
      LOG(FATAL) << "ERROR: " << strerror(errno);
    }
    buf += trans / sizeof(float);
    bytes_left -= trans;
  }

  close(fd);
}
}  // namespace offgs