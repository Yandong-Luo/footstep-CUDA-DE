### Function Test

#### Complie

```
nvcc -O3 -arch sm_86 -std=c++14 -Xcompiler -fPIC --use_fast_math test_sort.cu -o cuda_sort_test
```

#### Run

```
./cuda_sort_test
```

#### Compile big_E_F_matrix.cu
```
nvcc big_E_F_matrix.cu -o compute_big_E_F -lcublas -L/usr/local/cuda/lib64 -I/usr/local/cuda/include
```
#### Run

```
LD_LIBRARY_PATH=/usr/local/cuda/lib64 ./compute_big_E_F
```

#### Test sorting param
```
nvcc -O3 -arch sm_86 -std=c++14 -Xcompiler -fPIC --use_fast_math sort_param.cu -o sort_param
```
#### Run
```
./sort_param
```