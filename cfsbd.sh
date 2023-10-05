cd cFS
cp cfe/cmake/Makefile.sample Makefile
cp -r cfe/cmake/sample_defs sample_defs
make SIMULATION=native prep
make
make install
cd build/exe/cpu1/
./core-cpu1
