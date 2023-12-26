# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=write -ioengine=libaio \
#     -bs=2G -size=2G -numjobs=1 -group_reporting -name=init

####### write
# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=randwrite -ioengine=libaio \
#     -bs=16k -size=2G -numjobs=1 -group_reporting -name=randwrite16k





####### read (iodepth=1, numjobs=1)

# ### seq read
# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=read -ioengine=libaio \
#     -bs=4k -size=2G -numjobs=1 -group_reporting -name=seqread4k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=read -ioengine=libaio \
#     -bs=8k -size=2G -numjobs=1 -group_reporting -name=seqread8k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=read -ioengine=libaio \
#     -bs=16k -size=2G -numjobs=1 -group_reporting -name=seqread16k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=read -ioengine=libaio \
#     -bs=32k -size=2G -numjobs=1 -group_reporting -name=seqread32k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=read -ioengine=libaio \
#     -bs=64k -size=2G -numjobs=1 -group_reporting -name=seqread64k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=read -ioengine=libaio \
#     -bs=128k -size=2G -numjobs=1 -group_reporting -name=seqread128k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=read -ioengine=libaio \
#     -bs=256k -size=2G -numjobs=1 -group_reporting -name=seqread256k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=read -ioengine=libaio \
#     -bs=512k -size=2G -numjobs=1 -group_reporting -name=seqread512k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=read -ioengine=libaio \
#     -bs=1024k -size=2G -numjobs=1 -group_reporting -name=seqread1024k -runtime=30s

# ### rand read
# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=randread -ioengine=libaio \
#     -bs=4k -size=2G -numjobs=1 -group_reporting -name=randread4k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=randread -ioengine=libaio \
#     -bs=8k -size=2G -numjobs=1 -group_reporting -name=randread8k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=randread -ioengine=libaio \
#     -bs=16k -size=2G -numjobs=1 -group_reporting -name=randread16k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=randread -ioengine=libaio \
#     -bs=32k -size=2G -numjobs=1 -group_reporting -name=randread32k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=randread -ioengine=libaio \
#     -bs=64k -size=2G -numjobs=1 -group_reporting -name=randread64k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=randread -ioengine=libaio \
#     -bs=128k -size=2G -numjobs=1 -group_reporting -name=randread128k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=randread -ioengine=libaio \
#     -bs=256k -size=2G -numjobs=1 -group_reporting -name=randread256k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=randread -ioengine=libaio \
#     -bs=512k -size=2G -numjobs=1 -group_reporting -name=randread512k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=randread -ioengine=libaio \
#     -bs=1024k -size=2G -numjobs=1 -group_reporting -name=randread1024k -runtime=30s







####### read (iodepth=64, numjobs=1)

# ### seq read
# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=read -ioengine=libaio \
#     -bs=4k -size=2G -numjobs=1 -group_reporting -name=seqread4k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=read -ioengine=libaio \
#     -bs=8k -size=2G -numjobs=1 -group_reporting -name=seqread8k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=read -ioengine=libaio \
#     -bs=16k -size=2G -numjobs=1 -group_reporting -name=seqread16k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=read -ioengine=libaio \
#     -bs=32k -size=2G -numjobs=1 -group_reporting -name=seqread32k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=read -ioengine=libaio \
#     -bs=64k -size=2G -numjobs=1 -group_reporting -name=seqread64k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=read -ioengine=libaio \
#     -bs=128k -size=2G -numjobs=1 -group_reporting -name=seqread128k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=read -ioengine=libaio \
#     -bs=256k -size=2G -numjobs=1 -group_reporting -name=seqread256k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=read -ioengine=libaio \
#     -bs=512k -size=2G -numjobs=1 -group_reporting -name=seqread512k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=read -ioengine=libaio \
#     -bs=1024k -size=2G -numjobs=1 -group_reporting -name=seqread1024k -runtime=30s

# ### rand read
# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=randread -ioengine=libaio \
#     -bs=4k -size=2G -numjobs=1 -group_reporting -name=randread4k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=randread -ioengine=libaio \
#     -bs=8k -size=2G -numjobs=1 -group_reporting -name=randread8k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=randread -ioengine=libaio \
#     -bs=16k -size=2G -numjobs=1 -group_reporting -name=randread16k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=randread -ioengine=libaio \
#     -bs=32k -size=2G -numjobs=1 -group_reporting -name=randread32k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=randread -ioengine=libaio \
#     -bs=64k -size=2G -numjobs=1 -group_reporting -name=randread64k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=randread -ioengine=libaio \
#     -bs=128k -size=2G -numjobs=1 -group_reporting -name=randread128k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=randread -ioengine=libaio \
#     -bs=256k -size=2G -numjobs=1 -group_reporting -name=randread256k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=randread -ioengine=libaio \
#     -bs=512k -size=2G -numjobs=1 -group_reporting -name=randread512k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=randread -ioengine=libaio \
#     -bs=1024k -size=2G -numjobs=1 -group_reporting -name=randread1024k -runtime=30s






####### read (iodepth=1, numjobs=4)

### seq read
# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=read -ioengine=libaio \
#     -bs=4k -size=2G -numjobs=4 -group_reporting -name=seqread4k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=read -ioengine=libaio \
#     -bs=8k -size=2G -numjobs=4 -group_reporting -name=seqread8k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=read -ioengine=libaio \
#     -bs=16k -size=2G -numjobs=4 -group_reporting -name=seqread16k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=read -ioengine=libaio \
#     -bs=32k -size=2G -numjobs=4 -group_reporting -name=seqread32k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=read -ioengine=libaio \
#     -bs=64k -size=2G -numjobs=4 -group_reporting -name=seqread64k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=read -ioengine=libaio \
#     -bs=128k -size=2G -numjobs=4 -group_reporting -name=seqread128k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=read -ioengine=libaio \
#     -bs=256k -size=2G -numjobs=4 -group_reporting -name=seqread256k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=read -ioengine=libaio \
#     -bs=512k -size=2G -numjobs=4 -group_reporting -name=seqread512k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=read -ioengine=libaio \
#     -bs=1024k -size=2G -numjobs=4 -group_reporting -name=seqread1024k -runtime=30s

# ### rand read
# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=randread -ioengine=libaio \
#     -bs=4k -size=2G -numjobs=4 -group_reporting -name=randread4k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=randread -ioengine=libaio \
#     -bs=8k -size=2G -numjobs=4 -group_reporting -name=randread8k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=randread -ioengine=libaio \
#     -bs=16k -size=2G -numjobs=4 -group_reporting -name=randread16k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=randread -ioengine=libaio \
#     -bs=32k -size=2G -numjobs=4 -group_reporting -name=randread32k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=randread -ioengine=libaio \
#     -bs=64k -size=2G -numjobs=4 -group_reporting -name=randread64k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=randread -ioengine=libaio \
#     -bs=128k -size=2G -numjobs=4 -group_reporting -name=randread128k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=randread -ioengine=libaio \
#     -bs=256k -size=2G -numjobs=4 -group_reporting -name=randread256k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=randread -ioengine=libaio \
#     -bs=512k -size=2G -numjobs=4 -group_reporting -name=randread512k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=1 -thread -rw=randread -ioengine=libaio \
#     -bs=1024k -size=2G -numjobs=4 -group_reporting -name=randread1024k -runtime=30s






####### read (iodepth=64, numjobs=4)

# ### seq read
# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=read -ioengine=libaio \
#     -bs=4k -size=2G -numjobs=4 -group_reporting -name=seqread4k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=read -ioengine=libaio \
#     -bs=8k -size=2G -numjobs=4 -group_reporting -name=seqread8k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=read -ioengine=libaio \
#     -bs=16k -size=2G -numjobs=4 -group_reporting -name=seqread16k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=read -ioengine=libaio \
#     -bs=32k -size=2G -numjobs=4 -group_reporting -name=seqread32k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=read -ioengine=libaio \
#     -bs=64k -size=2G -numjobs=4 -group_reporting -name=seqread64k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=read -ioengine=libaio \
#     -bs=128k -size=2G -numjobs=4 -group_reporting -name=seqread128k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=read -ioengine=libaio \
#     -bs=256k -size=2G -numjobs=4 -group_reporting -name=seqread256k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=read -ioengine=libaio \
#     -bs=512k -size=2G -numjobs=4 -group_reporting -name=seqread512k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=read -ioengine=libaio \
#     -bs=1024k -size=2G -numjobs=4 -group_reporting -name=seqread1024k -runtime=30s

# ### rand read
# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=randread -ioengine=libaio \
#     -bs=4k -size=2G -numjobs=4 -group_reporting -name=randread4k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=randread -ioengine=libaio \
#     -bs=8k -size=2G -numjobs=4 -group_reporting -name=randread8k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=randread -ioengine=libaio \
#     -bs=16k -size=2G -numjobs=4 -group_reporting -name=randread16k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=randread -ioengine=libaio \
#     -bs=32k -size=2G -numjobs=4 -group_reporting -name=randread32k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=randread -ioengine=libaio \
#     -bs=64k -size=2G -numjobs=4 -group_reporting -name=randread64k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=randread -ioengine=libaio \
#     -bs=128k -size=2G -numjobs=4 -group_reporting -name=randread128k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=randread -ioengine=libaio \
#     -bs=256k -size=2G -numjobs=4 -group_reporting -name=randread256k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=randread -ioengine=libaio \
#     -bs=512k -size=2G -numjobs=4 -group_reporting -name=randread512k -runtime=30s

# sudo fio -filename=./temp/testfile -direct=1 -iodepth=64 -thread -rw=randread -ioengine=libaio \
#     -bs=1024k -size=2G -numjobs=4 -group_reporting -name=randread1024k -runtime=30s

fio -filename=/nvme2n1/fio_testfile -direct=1 -iodepth=512 -thread -rw=randread -ioengine=libaio -bs=64k -size=2G -numjobs=64 -group_reporting -name=randread64k -runtime=30s
