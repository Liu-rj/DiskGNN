for i in 1 2 4 8 16 32 128 256 16384 32768
# for i in 1 2 4 8 16 32 128 256
do
    # sudo sysctl -w vm.drop_caches=1
    # sleep 5
    ./a.out $i
done
