# Usage:

## Tensorflow/functionalTest.py
```python3 functionTest.py```

To check GPU passthrough and drivers are correctly installed and utilised.

## Tensorflow/train.py
```python3 train.py --log-file training.log```

Log file currently contains the metrics for epochs, timestamps and training details.

## Pytorch/functionalTest.py
```python3 functionTest.py```

## Pytorch/train.py
```python3 train.py --log-file training.log```

Log file currently contains the metrics for epochs, timestamps and training details.

## Pytorch (ROCm Docker Image)
1. Pull the ROCm image with Pytorch

```docker pull rocm/pytorch:latest```

2.  Docker Build

    ```docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd --device=/dev/dri --group-add video \`
    --ipc=host --shm-size 8G rocm/pytorch:latest
    ```

# Dockerfile Build in DIsconnected Environments
### Prerequisites
1. Cifar dataset in directory 'pytorch/data'
2. Latest rocm/pytorch docker image from 
```docker pull rocm/pytorch:latest```

### Steps to build
3. Build the image

    ```docker build -t rocmPytorch .```

4. Run the docker container
```docker run --rm -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
--device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 8G rocmPytorch
```

`cap-add=SYS_PTRACE` - for debugging, allows strace to attach to processes in the container

`security-opt seccomp=unconfined` - disables the seccomp security profile for unrestricted system calls

`device=/dev/kfd --device=/dev/dri`- passthrough gpu to container

`group-add video` - grant container permissions to the GPU 

`ipc=host` - shares the host's inter-process communication (IPC) namespace.

`shm-size 8G` - allocates 8GB of shared memory to the container


##
 Pytorch (ROCm Fresh Install)
```wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/torch-2.3.0%2Brocm6.2.3-cp310-cp310-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/torchvision-0.18.0%2Brocm6.2.3-cp310-cp310-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/pytorch_triton_rocm-2.3.0%2Brocm6.2.3.5a02332983-cp310-cp310-linux_x86_64.whl
pip3 uninstall torch torchvision pytorch-triton-rocm
pip3 install torch-2.3.0+rocm6.2.3-cp310-cp310-linux_x86_64.whl torchvision-0.18.0+rocm6.2.3-cp310-cp310-linux_x86_64.whl
pytorch_triton_rocm-2.3.0+rocm6.2.3.5a02332983-cp310-cp310-linux_x86_64.whl
```

```python3 -c 'import torch' 2> /dev/null && echo 'Success' || echo 'Failure'
python3 -c 'import torch; print(torch.cuda.is_available())'`
python3 -c "import torch; print(f'device name [0]:', torch.cuda.get_device_name(0))"
python3 -m torch.utils.collect_env
```

## ONNX runtime
1. Ensure MIGraphX is installed with the half library:

```dpkg -l | grep migraphx```
```dpkg -l | grep half```

2. Then install the ONNX runtime for ROCm.

```pip3 install onnxruntime-rocm -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/```

3. Verify:

```python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"```
