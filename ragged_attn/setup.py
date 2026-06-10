from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

# Check CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this package")

# Get CUDA version
cuda_version = torch.version.cuda
print(f"Building with CUDA {cuda_version}")

# Determine compute capabilities
def get_cuda_arch_flags():
    """Get appropriate CUDA architecture flags"""
    cuda_arch_list = []
    
    # Common architectures
    arch_dict = {
        '11.0': ['compute_70', 'sm_70', 'compute_75', 'sm_75', 'compute_80', 'sm_80', 'compute_86', 'sm_86'],
        '11.1': ['compute_70', 'sm_70', 'compute_75', 'sm_75', 'compute_80', 'sm_80', 'compute_86', 'sm_86'],
        '11.2': ['compute_70', 'sm_70', 'compute_75', 'sm_75', 'compute_80', 'sm_80', 'compute_86', 'sm_86'],
        '11.3': ['compute_70', 'sm_70', 'compute_75', 'sm_75', 'compute_80', 'sm_80', 'compute_86', 'sm_86'],
        '11.4': ['compute_70', 'sm_70', 'compute_75', 'sm_75', 'compute_80', 'sm_80', 'compute_86', 'sm_86'],
        '11.5': ['compute_70', 'sm_70', 'compute_75', 'sm_75', 'compute_80', 'sm_80', 'compute_86', 'sm_86'],
        '11.6': ['compute_70', 'sm_70', 'compute_75', 'sm_75', 'compute_80', 'sm_80', 'compute_86', 'sm_86'],
        '11.7': ['compute_70', 'sm_70', 'compute_75', 'sm_75', 'compute_80', 'sm_80', 'compute_86', 'sm_86'],
        '11.8': ['compute_70', 'sm_70', 'compute_75', 'sm_75', 'compute_80', 'sm_80', 'compute_86', 'sm_86'],
        '12.0': ['compute_70', 'sm_70', 'compute_75', 'sm_75', 'compute_80', 'sm_80', 'compute_86', 'sm_86', 'compute_89', 'sm_89'],
        '12.1': ['compute_70', 'sm_70', 'compute_75', 'sm_75', 'compute_80', 'sm_80', 'compute_86', 'sm_86', 'compute_89', 'sm_89'],
    }
    
    # Get architectures for current CUDA version
    if cuda_version in arch_dict:
        architectures = arch_dict[cuda_version]
    else:
        # Default to common architectures
        architectures = ['compute_70', 'sm_70', 'compute_75', 'sm_75', 'compute_80', 'sm_80', 'compute_86', 'sm_86']
    
    # Convert to gencode format
    gencode_flags = []
    for i in range(0, len(architectures), 2):
        if i + 1 < len(architectures):
            gencode_flags.append(f"-gencode=arch={architectures[i]},code={architectures[i+1]}")
    
    return gencode_flags

cxx_flags = [
    "-O3",
    "-std=c++17",  # <-- change this
    "-fPIC",
]

cuda_flags = [
    "-O3",
    # "-O0",
    # "-g",
    # "-G",
    "--use_fast_math",
    "-lineinfo",
    "--maxrregcount=255",
    "-Xptxas=-v",
    "-Xcompiler=-fPIC",
    "-std=c++17",  # <-- change this too
] + get_cuda_arch_flags()

# Include directories
include_dirs = [
    "include/",
]

# Define the extension
ext_modules = [
    CUDAExtension(
        name="ragged_multihead_attention",
        sources=[
            "attention_cuda.cu",
            "attention.cpp",
        ],
        include_dirs=include_dirs,
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": cuda_flags,
        },
        libraries=["cublas", "cublasLt", "curand"],
    ),
]

setup(
    name="variable_head_flash_attention",
    version="0.1.0",
    description="Optimized CUDA implementation of multi-head attention with variable head sizes",
    long_description=open("README.md", "r", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/variable-head-flash-attention",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=False),
    },
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
        ],
        "benchmark": [
            "matplotlib>=3.5.0",
            "pandas>=1.3.0",
            "seaborn>=0.11.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="attention, transformer, cuda, pytorch, optimization, flash-attention",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/variable-head-flash-attention/issues",
        "Source": "https://github.com/yourusername/variable-head-flash-attention",
    },
)