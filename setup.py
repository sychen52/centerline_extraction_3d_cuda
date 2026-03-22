from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="binary_thinning_3d",
    packages=["binary_thinning_3d"],
    install_requires=["torch", "numpy"],
    extras_require={"dev": ["SimpleITK", "itk-thickness3d"]},
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
