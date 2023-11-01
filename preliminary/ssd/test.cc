#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define ALIGNMENT 4096
#define file_path "/nvme2n1/test512"

uint64_t time_diff(struct timespec *start, struct timespec *end)
{
    return (end->tv_sec - start->tv_sec) * 1000000000 + (end->tv_nsec - start->tv_nsec);
}

void test_pread()
{
    std::ifstream is(file_path, std::ifstream::binary | std::ifstream::ate);
    std::size_t file_size = is.tellg();
    is.close();

    int fd = open(file_path, O_RDONLY);

    char *buf = (char *)malloc(file_size);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    if (pread(fd, buf, file_size, 0) == -1)
    {
        printf("error\n");
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    auto duration = time_diff(&start, &end) / 1000000.0;
    auto bandwidth = (file_size / (1024.0 * 1024.0 * 1024.0)) / (duration / 1000);
    uint64_t *array = (uint64_t *)buf;
    uint64_t sum = 0;
    for (int i = 0; i < file_size / sizeof(uint64_t); ++i)
    {
        sum += array[i];
    }
    printf("%lu, %fms, %fGB/s\n", sum, duration, bandwidth);
    free(buf);
    close(fd);
}

void test_pread_direct()
{
    std::ifstream is(file_path, std::ifstream::binary | std::ifstream::ate);
    std::size_t file_size = is.tellg();
    is.close();

    int fd = open(file_path, O_RDONLY | O_DIRECT);

    size_t reminder = file_size % ALIGNMENT;
    size_t aligned_size = reminder == 0 ? file_size : file_size - reminder + ALIGNMENT;
    char *buf = (char *)aligned_alloc(ALIGNMENT, aligned_size);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    if (pread(fd, buf, aligned_size, 0) == -1)
    {
        printf("error\n");
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    auto duration = time_diff(&start, &end) / 1000000.0;
    auto bandwidth = (file_size / (1024.0 * 1024.0 * 1024.0)) / (duration / 1000);
    uint64_t *array = (uint64_t *)buf;
    uint64_t sum = 0;
    for (int i = 0; i < file_size / sizeof(uint64_t); ++i)
    {
        sum += array[i];
    }
    printf("%lu, %fms, %fGB/s\n", sum, duration, bandwidth);
    free(buf);
    close(fd);
}

void test_pread_openmp(int i)
{
    std::ifstream is(file_path, std::ifstream::binary | std::ifstream::ate);
    std::size_t file_size = is.tellg();
    is.close();

    int fd = open(file_path, O_RDONLY);

    size_t block_size = ALIGNMENT * i;
    char *buf = (char *)malloc(file_size);
    auto num_blocks = static_cast<int>((file_size + block_size - 1) / block_size);
    std::cout << block_size / 1024 << "KB " << block_size / (1024 * 1024) << "MB" << std::endl;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

#pragma omp parallel for num_threads(64)
    for (int i = 0; i < num_blocks; ++i)
    {
        size_t read_size = i == (num_blocks - 1) ? file_size - i * block_size : block_size;
        if (pread(fd, buf + i * block_size, read_size, i * block_size) == -1)
        {
            printf("error\n");
        }
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    auto duration = time_diff(&start, &end) / 1000000.0;
    auto bandwidth = (file_size / (1024.0 * 1024.0 * 1024.0)) / (duration / 1000);
    uint64_t *array = (uint64_t *)buf;
    uint64_t sum = 0;
    for (int i = 0; i < file_size / sizeof(uint64_t); ++i)
    {
        sum += array[i];
    }
    printf("%lu, %fms, %fGB/s\n", sum, duration, bandwidth);
    free(buf);
    close(fd);
}

void test_pread_direct_openmp(int i)
{
    std::ifstream is(file_path, std::ifstream::binary | std::ifstream::ate);
    std::size_t file_size = is.tellg();
    is.close();

    int fd = open(file_path, O_RDONLY | O_DIRECT);

    size_t block_size = ALIGNMENT * i;
    size_t reminder = file_size % ALIGNMENT;
    size_t aligned_size = reminder == 0 ? file_size : file_size - reminder + ALIGNMENT;
    char *buf = (char *)aligned_alloc(ALIGNMENT, aligned_size);
    auto num_blocks = static_cast<int>((aligned_size + block_size - 1) / block_size);
    // auto omp_threads = std::min(num_blocks, 64);
    std::cout << block_size / 1024 << "KB " << block_size / (1024 * 1024) << "MB" << std::endl;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

#pragma omp parallel for num_threads(64)
    for (int i = 0; i < num_blocks; ++i)
    {
        size_t read_size = i == (num_blocks - 1) ? aligned_size - i * block_size : block_size;
        if (pread(fd, buf + i * block_size, read_size, i * block_size) == -1)
        {
            printf("error\n");
        }
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    auto duration = time_diff(&start, &end) / 1000000.0;
    auto bandwidth = (file_size / (1024.0 * 1024.0 * 1024.0)) / (duration / 1000);
    uint64_t *array = (uint64_t *)buf;
    uint64_t sum = 0;
    for (int i = 0; i < file_size / sizeof(uint64_t); ++i)
    {
        sum += array[i];
    }
    printf("%lu, %fms, %fGB/s\n", sum, duration, bandwidth);
    free(buf);
    close(fd);
}

void test_pread_direct_openmp_randread(int i)
{
    std::ifstream is(file_path, std::ifstream::binary | std::ifstream::ate);
    std::size_t file_size = is.tellg();
    is.close();

    int fd = open(file_path, O_RDONLY | O_DIRECT);

    int num_read = 2000;
    size_t block_size = ALIGNMENT * i;
    size_t buf_size = block_size * num_read;
    char *buf = (char *)aligned_alloc(ALIGNMENT, buf_size);
    std::cout << block_size / 1024 << "KB " << block_size / (1024 * 1024) << "MB" << std::endl;
    auto num_blocks = file_size / block_size;
    int offset[num_read];
    for (int i = 0; i < num_read; i++)
    {
        offset[i] = rand() % num_blocks;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

#pragma omp parallel for num_threads(64)
    for (int i = 0; i < num_read; ++i)
    {
        if (pread(fd, buf + i * block_size, block_size, offset[i] * block_size) == -1)
        {
            printf("error\n");
        }
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    auto duration = time_diff(&start, &end) / 1000000.0;
    auto bandwidth = (buf_size / (1024.0 * 1024.0 * 1024.0)) / (duration / 1000);
    uint64_t *array = (uint64_t *)buf;
    uint64_t sum = 0;
    for (int i = 0; i < buf_size / sizeof(uint64_t); ++i)
    {
        sum += array[i];
    }
    printf("%lu, %fms, %fGB/s\n", sum, duration, bandwidth);
    free(buf);
    close(fd);
}

int main(int argc, char *argv[])
{
    int i = atoi(argv[1]);
    test_pread_direct();
}