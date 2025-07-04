import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
os.environ["CC"] = "/usr/local/gcc-8.3.0/bin/gcc"
os.environ["CXX"] = "/usr/local/gcc-8.3.0/bin/g++"
setup(
    name='iou3d',
    ext_modules=[
        CUDAExtension('iou3d_cuda', [
            'src/iou3d.cpp',
            'src/iou3d_kernel.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})
