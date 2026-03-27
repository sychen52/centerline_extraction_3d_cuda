import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Read README.md for long_description
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="binary_thinning_3d",
    version="1.0.8",
    author="Shiyang Chen",
    author_email="sychen52@gmail.com",
    description="A fast 3D binary thinning implementation using CUDA and PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sychen52/binary_thinning_3d_cuda",
    project_urls={
        "Bug Tracker": "https://github.com/sychen52/binary_thinning_3d_cuda/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Intended Audience :: Science/Research",
    ],
    packages=["binary_thinning_3d"],
    install_requires=["torch", "numpy"],
    extras_require={"dev": ["SimpleITK", "itk-thickness3d"]},
    python_requires=">=3.8",
    ext_modules=[
        CUDAExtension(
            "binary_thinning_3d.cuda_thinning_ext",
            [
                "csrc/thinning.cpp",
                "csrc/thinning_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
