set -exu

mkdir -p build

docker run \
    --rm -it -v $(pwd):$(pwd) -w $(pwd) \
    nvidia/cuda:11.7.0-devel-ubuntu20.04 \
        nvcc \
            mvncdf_kernel.cu \
            practice/host.cu \
            -o ./build/output.out

# nvprof

docker run \
    --rm -it -v $(pwd):$(pwd) -w $(pwd) \
    nvidia/cuda:11.7.0-devel-ubuntu20.04 \
        nvcc \
            mvncdf_host.cu \
            -o ./build/output-mvncdf.out
