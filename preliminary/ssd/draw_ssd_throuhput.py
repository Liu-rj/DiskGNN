import numpy as np
import matplotlib.pyplot as plt

seqread_iodepth1_numjobs1 = [7.82, 15.2, 30.1, 56.2, 88.0, 140, 137, 138, 138]
seqread_iodepth1_numjobs4 = [12.7, 25.2, 50.4, 122, 135, 135, 135, 135, 135]
seqread_iodepth64_numjobs1 = [12.7, 25.2, 50.3, 140, 138, 137, 137, 138, 137]
seqread_iodepth64_numjobs4 = [12.7, 25.2, 50.4, 136, 134, 135, 135, 135, 135]

randread_iodepth1_numjobs1 = [7.51, 15.5, 31.0, 55.7, 102, 140, 137, 137, 137]
randread_iodepth1_numjobs4 = [12.7, 25.2, 50.4, 101, 136, 135, 135, 135, 135]
randread_iodepth64_numjobs1 = [12.7, 25.2, 50.4, 102, 140, 137, 138, 138, 138]
randread_iodepth64_numjobs4 = [12.7, 25.2, 50.4, 101, 136, 135, 135, 135, 135]


block_size = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
xticks = [4, 32, 64, 128, 256, 512, 1024]
font_size = 10


##### sequential read
plt.figure(figsize=(10, 5))
plt.plot(block_size, seqread_iodepth1_numjobs1, label="iodepth=1,numjobs=1", marker=".")
plt.plot(block_size, seqread_iodepth1_numjobs4, label="iodepth=1,numjobs=4", marker="v")
plt.plot(
    block_size, seqread_iodepth64_numjobs1, label="iodepth=64,numjobs=1", marker="1"
)
plt.plot(
    block_size, seqread_iodepth64_numjobs4, label="iodepth=64,numjobs=4", marker="*"
)
plt.xticks(xticks, fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.xlabel(
    "block size (KB)",
    fontsize=font_size,
    fontweight="bold",
)
plt.ylabel("throughput (MB/s)", fontsize=font_size, fontweight="bold")
plt.title("SSD Sequential Read Throughput", fontsize=font_size, fontweight="bold")
plt.grid(linestyle="-.")
plt.legend(
    fontsize=font_size,
    edgecolor="k",
)
plt.savefig("SSD-seqread-throughput.pdf", bbox_inches="tight")
plt.clf()


##### random read
plt.figure(figsize=(10, 5))
plt.plot(
    block_size, randread_iodepth1_numjobs1, label="iodepth=1,numjobs=1", marker="."
)
plt.plot(
    block_size, randread_iodepth1_numjobs4, label="iodepth=1,numjobs=4", marker="v"
)
plt.plot(
    block_size, randread_iodepth64_numjobs1, label="iodepth=64,numjobs=1", marker="1"
)
plt.plot(
    block_size, randread_iodepth64_numjobs4, label="iodepth=64,numjobs=4", marker="*"
)
plt.xticks(xticks, fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.xlabel(
    "block size (KB)",
    fontsize=font_size,
    fontweight="bold",
)
plt.ylabel("throughput (MB/s)", fontsize=font_size, fontweight="bold")
plt.title("SSD Random Read Throughput", fontsize=font_size, fontweight="bold")
plt.grid(linestyle="-.")
plt.legend(
    fontsize=font_size,
    edgecolor="k",
)
plt.savefig("SSD-randread-throughput.pdf", bbox_inches="tight")
plt.clf()


##### both seqread and randread
font_size = 15
plt.figure(figsize=(20, 5))
# plt.plot(
#     block_size, seqread_iodepth1_numjobs1, label="seq,iodepth=1,numjobs=1", marker="v"
# )
plt.plot(
    block_size, seqread_iodepth1_numjobs4, label="seq,iodepth=1,numjobs=4", marker="^"
)
plt.plot(
    block_size, seqread_iodepth64_numjobs1, label="seq,iodepth=64,numjobs=1", marker="<"
)
plt.plot(
    block_size, seqread_iodepth64_numjobs4, label="seq,iodepth=64,numjobs=4", marker=">"
)
# plt.plot(
#     block_size, randread_iodepth1_numjobs1, label="rand,iodepth=1,numjobs=1", marker="1"
# )
plt.plot(
    block_size, randread_iodepth1_numjobs4, label="rand,iodepth=1,numjobs=4", marker="2"
)
plt.plot(
    block_size,
    randread_iodepth64_numjobs1,
    label="rand,iodepth=64,numjobs=1",
    marker="3",
)
plt.plot(
    block_size,
    randread_iodepth64_numjobs4,
    label="rand,iodepth=64,numjobs=4",
    marker="4",
)
plt.xticks(xticks, fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.xlabel(
    "block size (KB)",
    fontsize=font_size,
    fontweight="bold",
)
plt.ylabel("throughput (MB/s)", fontsize=font_size, fontweight="bold")
plt.title("SSD Read Throughput", fontsize=font_size, fontweight="bold")
plt.grid(linestyle="-.")
plt.legend(fontsize=font_size, edgecolor="k", ncols=2)
plt.savefig("SSD-throughput.pdf", bbox_inches="tight")
plt.clf()
