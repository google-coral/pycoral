# Copyright 2019-2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

config_setting(
    name = "windows",
    values = {
        "cpu": "x64_windows",
    },
)

cc_library(
    name = "builddata",
    linkstamp = select({
        ":windows": None,  # Linkstamp doesn't work on Windows, https://github.com/bazelbuild/bazel/issues/6997
        "//conditions:default": "builddata.cc",
    }),
)

genrule(
    name = "rc_tpl",
    srcs = [
        "builddata.cc",
        "edgetpu.rc.tpl",
        "edgetpu_rc.ps1",
    ],
    outs = [
        "edgetpu.rc",
    ],
    cmd_ps = "$(location edgetpu_rc.ps1) " +
             "-BuildDataFile $(location builddata.cc) " +
             "-BuildStatus bazel-out\\stable-status.txt " +
             "-ResFileTemplate $(location edgetpu.rc.tpl) " +
             "-OutputFile $(location edgetpu.rc)",
    stamp = 1,
)

genrule(
    name = "dll_res_gen",
    srcs = [
        "edgetpu.rc",
    ],
    outs = [
        "edgetpu.res",
    ],
    cmd_bat = "rc.exe /nologo /fo $(location edgetpu.res) $(location edgetpu.rc)",
)

pybind_extension(
    name = "_pywrap_coral",
    srcs = [
        "coral_wrapper.cc",
    ],
    hdrs = [],
    data = select({
        ":windows": [":edgetpu.res"],
        "//conditions:default": [],
    }),
    linkopts = select({
        ":windows": ["$(location :edgetpu.res)"],
        "//conditions:default": [],
    }),
    module_name = "_pywrap_coral",
    deps = [
        ":builddata",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@libcoral//coral:bbox",
        "@libcoral//coral:tflite_utils",
        "@libcoral//coral/learn:imprinting_engine",
        "@libcoral//coral/learn:utils",
        "@libcoral//coral/learn/backprop:softmax_regression_model",
        "@libcoral//coral/pipeline:allocator",
        "@libcoral//coral/pipeline:common",
        "@libcoral//coral/pipeline:pipelined_model_runner",
        "@libedgetpu//tflite/public:edgetpu",
        "@local_config_python//:numpy_headers",
        "@local_config_python//:python_headers",
        "@org_tensorflow//tensorflow/lite:stateful_error_reporter",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@pybind11",
    ],
)
