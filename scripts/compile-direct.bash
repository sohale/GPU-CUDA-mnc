set -exu

mkdir -p build

nvcc \
            practice/host.cu \
            -o ./build/output-d.out
