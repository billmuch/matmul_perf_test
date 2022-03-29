#!/bin/bash

clang++ blis.cpp -DFLOAT32 -fopenmp -g -O2 -o blis_float32.x -I/home/neal/local/include/blis -L/home/neal/local/lib -lblis 
clang++ blis.cpp -DFLOAT64 -fopenmp -g -O2 -o blis_float64.x -I/home/neal/local/include/blis -L/home/neal/local/lib -lblis 