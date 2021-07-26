# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
workspace(name = "pycoral")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Configure libedgetpu and downstream libraries (TF and Crosstool).
local_repository(
    name = "libedgetpu",
    path = "libedgetpu",
)

load("@libedgetpu//:workspace.bzl", "libedgetpu_dependencies")
libedgetpu_dependencies()

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")
tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")
tf_workspace0()

load("@coral_crosstool//:configure.bzl", "cc_crosstool")
cc_crosstool(name = "crosstool", cpp_version = "c++14")

http_archive(
    name = "com_google_glog",
    sha256 = "6fc352c434018b11ad312cd3b56be3597b4c6b88480f7bd4e18b3a3b2cf961aa",
    strip_prefix = "glog-3ba8976592274bc1f907c402ce22558011d6fc5e",
    urls = [
        "https://github.com/google/glog/archive/3ba8976592274bc1f907c402ce22558011d6fc5e.tar.gz",
    ],
    build_file_content = """
licenses(['notice'])
exports_files(['CMakeLists.txt'])
load(':bazel/glog.bzl', 'glog_library')
glog_library(with_gflags=0)
""",
)

local_repository(
    name = "libcoral",
    path = "libcoral",
)

new_local_repository(
    name = "glog",
    path = "libcoral/third_party/glog",
    build_file = "libcoral/third_party/glog/BUILD",
)

load("@org_tensorflow//third_party/py:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")
