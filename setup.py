# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import atexit
import os
import sys
import subprocess
import io
import re
import copy
import tempfile
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
from distutils.file_util import copy_file

path = os.path.dirname(os.path.realpath(__file__))

CUDA_HOME = "/usr/local/cuda"
os.environ["CUDA_HOME"] = CUDA_HOME

with open(path + "/VERSION", "r") as f:
    te_version = f.readline()

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]
    return raw_output, bare_metal_major, bare_metal_minor


def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
    if int(bare_metal_major) >= 11 and int(bare_metal_minor) >= 2:
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args


def extra_gencodes(cc_flag):
    _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
    if int(bare_metal_major) >= 11:
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
        if int(bare_metal_minor) >= 8:
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")


def extra_compiler_flags():
    return [
        "-O3",
        "-gencode",
        "arch=compute_70,code=sm_70",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "-I./transformer_engine/common/layer_norm/",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ]


cc_flag = []
extra_gencodes(cc_flag)


def make_abs_path(l):
    return [os.path.join(path, p) for p in l]


include_dirs = [
    "transformer_engine/common/include",
]
include_dirs = make_abs_path(include_dirs)

tensorflow_sources = [
    "transformer_engine/tensorflow/src/pywrap_transformer_engine.cc",
]
tensorflow_sources = make_abs_path(tensorflow_sources)

all_sources = tensorflow_sources

supported_frameworks = {
    "all": all_sources,
    "tensorflow": tensorflow_sources,
}

framework = "all"

args = sys.argv.copy()
for s in args:
    if s.startswith("--framework="):
        framework = s.replace("--framework=", "")
        sys.argv.remove(s)

if framework not in supported_frameworks.keys():
    raise ValueError("Unsupported framework " + framework)


class CMakeExtension(Extension):
    def __init__(self, name, cmake_path, sources, **kwargs):
        super(CMakeExtension, self).__init__(name, sources=sources, **kwargs)
        self.cmake_path = cmake_path


ext_modules = []

ext_modules.append(
    CMakeExtension(
        name="transformer_engine",
        cmake_path=os.path.join(path, "transformer_engine"),
        sources=[],
        include_dirs=include_dirs,
    )
)

def get_cmake_bin():
    cmake_bin = "cmake"
    try:
        out = subprocess.check_output([cmake_bin, "--version"])
    except OSError:
        cmake_installed_version = LooseVersion("0.0")
    else:
        cmake_installed_version = LooseVersion(
            re.search(r"version\s*([\d.]+)", out.decode()).group(1)
        )

    if cmake_installed_version < LooseVersion("3.18.0"):
        print(
            "Could not find a recent CMake to build Transformer Engine. "
            "Attempting to install CMake 3.18 to a temporary location via pip.",
            flush=True,
        )
        cmake_temp_dir = tempfile.TemporaryDirectory(prefix="nvte-cmake-tmp")
        atexit.register(cmake_temp_dir.cleanup)
        try:
            _ = subprocess.check_output(
                ["pip", "install", "--target", cmake_temp_dir.name, "cmake~=3.18.0"]
            )
        except Exception:
            raise RuntimeError(
                "Failed to install temporary CMake. "
                "Please update your CMake to 3.18+."
            )
        cmake_bin = os.path.join(cmake_temp_dir.name, "bin", "run_cmake")
        with io.open(cmake_bin, "w") as f_run_cmake:
            f_run_cmake.write(
                f"#!/bin/sh\nPYTHONPATH={cmake_temp_dir.name} {os.path.join(cmake_temp_dir.name, 'bin', 'cmake')} \"$@\""
            )
        os.chmod(cmake_bin, 0o755)

    return cmake_bin


class CMakeBuildExtension(build_ext, object):
    def __init__(self, *args, **kwargs) -> None:
        super(CMakeBuildExtension, self).__init__(*args, **kwargs)

    def build_extensions(self) -> None:
        print("Building CMake extensions!")

        cmake_bin = get_cmake_bin()
        config = "Debug" if self.debug else "Release"

        ext_name = self.extensions[0].name
        build_dir = self.get_ext_fullpath(ext_name).replace(
            self.get_ext_filename(ext_name), ""
        )
        build_dir = os.path.abspath(build_dir)

        cmake_args = [
            "-DCMAKE_BUILD_TYPE=" + config,
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(config.upper(), build_dir),
            "-DCMAKE_CUDA_ARCHITECTURES=90",
        ]

        cmake_build_args = ["--config", config]

        cmake_build_dir = os.path.join(self.build_temp, config)
        if not os.path.exists(cmake_build_dir):
            os.makedirs(cmake_build_dir)

        config_and_build_commands = [
            [cmake_bin, self.extensions[0].cmake_path] + cmake_args,
            [cmake_bin, "--build", "."] + cmake_build_args,
        ]

        if True:
            print(f"Running CMake in {cmake_build_dir}:")
            for command in config_and_build_commands:
                print(" ".join(command))
            sys.stdout.flush()

        # Config and build the extension
        try:
            for command in config_and_build_commands:
                subprocess.check_call(command, cwd=cmake_build_dir)
        except OSError as e:
            raise RuntimeError("CMake failed: {}".format(str(e)))


setup(
    name="transformer_engine",
    version=te_version,
    packages=find_packages(
        exclude=(
            "build",
            "src",
            "include",
            "tests",
            "dist",
            "docs",
            "tests",
            "examples",
            "transformer_engine.egg-info",
        )
    ),
    description="Transformer acceleration library",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuildExtension},
    license_files=("LICENSE",),
    install_requires=["pybind11", "pydantic"],
)
