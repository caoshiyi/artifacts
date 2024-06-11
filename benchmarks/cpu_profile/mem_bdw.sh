# download likwid-bench
# VERSION=stable
# wget http://ftp.fau.de/pub/likwid/likwid-$VERSION.tar.gz
# tar -xaf likwid-$VERSION.tar.gz
# cd likwid-*
# vi config.mk # configure build, e.g. change installation prefix and architecture flags
# make
# sudo make install

# mem bdw
likwid-bench -t load_mem -w S0:4GB
likwid-bench -t store_mem -w S0:4GB