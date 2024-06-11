# download likwid-bench
# VERSION=stable
# wget http://ftp.fau.de/pub/likwid/likwid-$VERSION.tar.gz
# tar -xaf likwid-$VERSION.tar.gz
# cd likwid-*
# vi config.mk # configure build, e.g. change installation prefix and architecture flags
# make
# sudo make install

# flops if avx512 supported: peakflops_sp_avx512_fma
likwid-bench -t peakflops_sp_avx_fma -W N:2GB:3