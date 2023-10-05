cd algos
gcc wait.c -o waitc -w
gcc C_MAIN.c -o mainc -w
gcc MEmaze.c -o MEmazec -w
cd MEmaze
make
cd ..
gcc MEpuzzle.c -o MEpuzzlec -w
cd MEpuzzle
make
cd ..
gcc MEsort.c -o MEsortc -w
cd MEsort
make
cd ..
gcc MEopenmp.c -o MEopenmp_c -w
gcc -fopenmp MEopenmp/0openMP_HiWor.c -o MEopenmp/openMP0 -w
gcc -fopenmp MEopenmp/1openMP_forNW.c -o MEopenmp/openMP1 -w
gcc -fopenmp MEopenmp/2openMP_forNWB.c -o MEopenmp/openMP2 -w
gcc -fopenmp MEopenmp/3openMP_Sched.c -o MEopenmp/openMP3 -w
gcc -fopenmp MEopenmp/4openMP_atomic.c -o MEopenmp/openMP4 -w
gcc -fopenmp MEopenmp/5openMP_Thrdpriv.c -o MEopenmp/openMP5 -w
gcc -fopenmp MEopenmp/6openMP_flush.c -o MEopenmp/openMP6 -w
gcc -fopenmp MEopenmp/7openMP_nested.c -o MEopenmp/openMP7 -w
gcc MEcuda.c -o MEcuda_c -w
nvcc MEcuda/0cuda_HiWor.cu -o MEcuda/MEcuda0 -w
