from setuptools import setup, find_packages, Extension
import os
import torch
import torch.utils.cpp_extension as torch_cpp_ext

ext_modules = []
ext_modules.append(
    Extension(
        name="fastmoe._cpu_kernel",
        sources=["fastmoe/csrc/flashattention.cpp"],
        include_dirs=torch_cpp_ext.include_paths(),
        library_dirs=torch.utils.cpp_extension.library_paths(),
        libraries=['torch', 'c10', 'torch_cpu', 'torch_python', 'mkl_rt'],
        language="c++",
        # extra_compile_args=["-std=c++17", "-O1", "-fopenmp", "-mavx2", "-mfma", "-mf16c", "-Wno-ignored-qualifiers"],
        extra_compile_args=["-std=c++17", "-O1", "-fopenmp", "-mavx512f", "-mavx512cd", "-mavx512vl", "-mf16c", "-Wno-ignored-qualifiers"],
        extra_link_args=['-lpthread', '-lm', '-ldl']
    )
)

setup(
    name="FastMoE",
    version="0.0.1",
    packages=find_packages(
        exclude=("build", "include", "csrc", "test", "traces", "notebooks", "benchmarks", "fastmoe.egg-info")
    ),
    author="model toolchain",
    author_email="",
    description="Efficient MoE Inference",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
    python_requires=">=3.9",
    install_requires=[
        "aiohttp",
        "fastapi",
        "zmq",
        "vllm>=0.2.7",
        "rpyc",
        "torch==2.1.2",
        "uvloop",
        "uvicorn",
        "psutil",
        "interegular",
        "lark",
        "numba",
        "pydantic",
        "referencing",
        "diskcache", 
        "cloudpickle",
        "pillow",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": torch_cpp_ext.BuildExtension},
)