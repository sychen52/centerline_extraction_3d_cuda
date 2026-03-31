import os
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

# Read README.md for long_description
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Check if CUDA is available. If not, we are likely building a source distribution (sdist)
# or just gathering metadata, so we use a dummy extension.
if CUDA_HOME is None:
    # Use a dummy extension so metadata can be gathered without CUDA checks
    ext_modules = [
        Extension(
            "binary_thinning_3d.cuda_thinning_ext",
            ["csrc/thinning.cpp", "csrc/thinning_kernel.cu"],
        )
    ]
else:
    # Use the real CUDAExtension for binary wheel builds
    ext_modules = [
        CUDAExtension(
            "binary_thinning_3d.cuda_thinning_ext",
            [
                "csrc/thinning.cpp",
                "csrc/thinning_kernel.cu",
            ],
        ),
    ]

setup(
    name="binary_thinning_3d_cuda",
    version="1.2.1",
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
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
