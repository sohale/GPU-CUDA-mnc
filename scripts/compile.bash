set -exu

mkdir -p build

docker run -it -v $(pwd):$(pwd) -w $(pwd) nvidia/cuda:11.7.0-devel-ubuntu20.04 \
    nvcc \
    cdf_kernel.cu \
    practice/host.cu \
    -o ./build/output.out
