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
SHELL := /bin/bash
PYTHON ?= $(shell which python3)
MAKEFILE_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
PY3_VER ?= $(shell $(PYTHON) -c "import sys;print('%d%d' % sys.version_info[:2])")
OS := $(shell uname -s)
ARCH := $(shell uname -p)

ifeq ($(PYTHON),)
$(error PYTHON must be set)
endif

ifeq ($(OS),Linux)
CPU ?= k8
else ifeq ($(OS),Darwin)
ifeq ($(ARCH),arm)
CPU ?= darwin_arm64
DARWIN_CPU := darwin_arm64
else
CPU ?= darwin
DARWIN_CPU := darwin_x86_64
endif
else
$(error $(OS) is not supported)
endif

ifeq ($(filter $(CPU),k8 armv7a aarch64 darwin darwin_arm64),)
$(error CPU must be k8, armv7a, aarch64, darwin, or darwin_arm64)
endif

# Allowed COMPILATION_MODE values: opt, dbg, fastbuild
COMPILATION_MODE ?= opt
ifeq ($(filter $(COMPILATION_MODE),opt dbg fastbuild),)
$(error COMPILATION_MODE must be opt, dbg or fastbuild)
endif

BAZEL_OUT_DIR :=  $(MAKEFILE_DIR)/bazel-out/$(CPU)-$(COMPILATION_MODE)/bin
COMMON_BAZEL_BUILD_FLAGS := --compilation_mode=$(COMPILATION_MODE) \
                            --copt=-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
                            --cpu=$(CPU)

BAZEL_BUILD_FLAGS_Linux := --linkopt=-L$(MAKEFILE_DIR)/libedgetpu_bin/direct/$(CPU) \
                           --linkopt=-l:libedgetpu.so.1
BAZEL_BUILD_FLAGS_Darwin := --linkopt=-L$(MAKEFILE_DIR)/libedgetpu_bin/direct/$(DARWIN_CPU) \
                            --linkopt=-ledgetpu.1

ifeq ($(COMPILATION_MODE), opt)
BAZEL_BUILD_FLAGS_Linux += --linkopt=-Wl,--strip-all
endif

# Extension naming conventions changed since python 3.8
# https://docs.python.org/3/whatsnew/3.8.html#build-and-c-api-changes
ifeq ($(shell test $(PY3_VER) -ge 38; echo $$?),0)
PY3_VER_EXT=$(PY3_VER)
else
PY3_VER_EXT=$(PY3_VER)m
endif

ifeq ($(CPU),k8)
PY_WRAPPER_SUFFIX := x86_64-linux-gnu.so
PY_DIST_PLATFORM := linux_x86_64
else ifeq ($(CPU),aarch64)
BAZEL_BUILD_FLAGS_Linux += --copt=-ffp-contract=off
PY_WRAPPER_SUFFIX := aarch64-linux-gnu.so
PY_DIST_PLATFORM := linux_aarch64
else ifeq ($(CPU),armv7a)
BAZEL_BUILD_FLAGS_Linux += --copt=-ffp-contract=off
PY_WRAPPER_SUFFIX := arm-linux-gnueabihf.so
PY_DIST_PLATFORM := linux-armv7l
else ifeq ($(findstring darwin,$(CPU)),darwin)
PY_WRAPPER_SUFFIX := darwin.so
endif

CORAL_WRAPPER_NAME := _pywrap_coral.cpython-$(PY3_VER_EXT)-$(PY_WRAPPER_SUFFIX)
TFLITE_WRAPPER_NAME := _pywrap_tensorflow_interpreter_wrapper.cpython-$(PY3_VER_EXT)-$(PY_WRAPPER_SUFFIX)
TFLITE_WRAPPER_TARGET := @org_tensorflow//tensorflow/lite/python/interpreter_wrapper:_pywrap_tensorflow_interpreter_wrapper

BAZEL_BUILD_FLAGS := $(COMMON_BAZEL_BUILD_FLAGS) \
                     $(BAZEL_BUILD_FLAGS_$(OS))

TFLITE_BAZEL_BUILD_FLAGS := $(COMMON_BAZEL_BUILD_FLAGS)

CORAL_WRAPPER_OUT_DIR  := $(MAKEFILE_DIR)/pycoral/pybind
TFLITE_WRAPPER_OUT_DIR := $(MAKEFILE_DIR)/tflite_runtime

TENSORFLOW_DIR = $(shell bazel info output_base)/external/org_tensorflow/
TENSORFLOW_VERSION_SUFFIX := .post1
TENSORFLOW_VERSION = $(shell bazel aquery $(TFLITE_BAZEL_BUILD_FLAGS) $(TFLITE_WRAPPER_TARGET) >> /dev/null && \
                       grep "_VERSION = " "${TENSORFLOW_DIR}/tensorflow/tools/pip_package/setup.py" | cut -d= -f2 | sed "s/[ '-]//g")$(TENSORFLOW_VERSION_SUFFIX)
TENSORFLOW_COMMIT = $(shell bazel query "@libedgetpu_properties//..." | grep tensorflow_commit | cut -d\# -f2)

TFLITE_RUNTIME_VERSION = $(TENSORFLOW_VERSION)
TFLITE_RUNTIME_DIR := /tmp/tflite_runtime_root

# $(1): Package version
# $(2): Wrapper files
define prepare_tflite_runtime
rm -rf $(TFLITE_RUNTIME_DIR) && mkdir -p $(TFLITE_RUNTIME_DIR)/tflite_runtime
echo "__version__ = '$(1)'" \
     >> $(TFLITE_RUNTIME_DIR)/tflite_runtime/__init__.py
echo "__git_version__ = '$(TENSORFLOW_COMMIT)'" \
     >> $(TFLITE_RUNTIME_DIR)/tflite_runtime/__init__.py
cp $(MAKEFILE_DIR)/tflite_runtime/$(2) \
   $(TENSORFLOW_DIR)/tensorflow/lite/python/{interpreter.py,metrics_interface.py,metrics_portable.py} \
   $(TFLITE_RUNTIME_DIR)/tflite_runtime/
sed -e "s/'numpy.*/'numpy>=1.16.0',/" $(TENSORFLOW_DIR)/tensorflow/lite/tools/pip_package/setup_with_binary.py \
    > $(TFLITE_RUNTIME_DIR)/setup.py
endef

# $(1): Package version
define prepare_tflite_runtime_debian
cp -r $(TENSORFLOW_DIR)/tensorflow/lite/tools/pip_package/debian \
      $(TFLITE_RUNTIME_DIR)
sed -e "s/pycoral\/pybind\/_pywrap_coral/tflite_runtime\/_pywrap_tensorflow_interpreter_wrapper/" \
    -e "s/pycoral/tflite_runtime/" \
    $(MAKEFILE_DIR)/debian/rules \
    > $(TFLITE_RUNTIME_DIR)/debian/rules
echo -e "tflite-runtime ($(1)) unstable; urgency=low\
\n\
\n  * Bump version to $(1).\
\n\
\n -- TensorFlow team <packages@tensorflow.org>  $(shell date -R)" \
    > $(TFLITE_RUNTIME_DIR)/debian/changelog
endef

.PHONY: all \
        pybind \
        tflite \
        clean \
        deb \
        wheel \
        wheel-all \
        tflite-wheel \
        tflite-deb \
        help

all: pybind tflite

pybind:
	PYTHON_BIN_PATH=$(PYTHON) bazel build $(BAZEL_BUILD_FLAGS) \
	    --embed_label='TENSORFLOW_COMMIT=$(TENSORFLOW_COMMIT)' \
	    --stamp \
	    //src:_pywrap_coral
	mkdir -p $(CORAL_WRAPPER_OUT_DIR)
	touch $(CORAL_WRAPPER_OUT_DIR)/__init__.py
	cp -f $(BAZEL_OUT_DIR)/src/_pywrap_coral.so $(CORAL_WRAPPER_OUT_DIR)/$(CORAL_WRAPPER_NAME)

tflite:
	PYTHON_BIN_PATH=$(PYTHON) bazel build $(TFLITE_BAZEL_BUILD_FLAGS) $(TFLITE_WRAPPER_TARGET)
	mkdir -p $(TFLITE_WRAPPER_OUT_DIR)
	cp -f $(BAZEL_OUT_DIR)/external/org_tensorflow/tensorflow/lite/python/interpreter_wrapper/_pywrap_tensorflow_interpreter_wrapper.so \
	      $(TFLITE_WRAPPER_OUT_DIR)/$(TFLITE_WRAPPER_NAME)
	cp -f $(TENSORFLOW_DIR)/tensorflow/lite/python/{interpreter.py,metrics_interface.py,metrics_portable.py} \
	      $(TFLITE_WRAPPER_OUT_DIR)
	touch $(TFLITE_WRAPPER_OUT_DIR)/__init__.py

clean:
	rm -rf $(MAKEFILE_DIR)/bazel-* \
	       $(MAKEFILE_DIR)/build \
	       $(MAKEFILE_DIR)/dist \
	       $(MAKEFILE_DIR)/*.egg-info \
	       $(MAKEFILE_DIR)/pycoral/pybind/*.so \
	       $(MAKEFILE_DIR)/tflite_runtime

deb:
	dpkg-buildpackage -rfakeroot -us -uc -tc -b
	dpkg-buildpackage -rfakeroot -us -uc -tc -b -a armhf -d
	dpkg-buildpackage -rfakeroot -us -uc -tc -b -a arm64 -d
	mkdir -p $(MAKEFILE_DIR)/dist
	mv $(MAKEFILE_DIR)/../*.{deb,changes,buildinfo} \
	   $(MAKEFILE_DIR)/dist

wheel: pybind
	rm -rf $(MAKEFILE_DIR)/*.egg-info
	WRAPPER_NAME=$(CORAL_WRAPPER_NAME) $(PYTHON) $(MAKEFILE_DIR)/setup.py \
	    clean --all \
	    bdist_wheel $(if $(PY_DIST_PLATFORM),--plat-name $(PY_DIST_PLATFORM),) \
	                -d $(MAKEFILE_DIR)/dist

wheel-all:
	rm -rf $(MAKEFILE_DIR)/*.egg-info
	$(PYTHON) $(MAKEFILE_DIR)/setup.py \
	    clean --all \
	    bdist_wheel -d $(MAKEFILE_DIR)/dist

tflite-wheel: tflite
	$(call prepare_tflite_runtime,$(TFLITE_RUNTIME_VERSION),$(TFLITE_WRAPPER_NAME))
	(cd $(TFLITE_RUNTIME_DIR) && \
	 export PACKAGE_VERSION=$(TFLITE_RUNTIME_VERSION) && \
	 $(PYTHON) setup.py \
	     clean --all \
	     bdist_wheel $(if $(PY_DIST_PLATFORM),--plat-name $(PY_DIST_PLATFORM),) \
	     -d $(MAKEFILE_DIR)/dist)

tflite-deb:
	$(call prepare_tflite_runtime,$(TFLITE_RUNTIME_VERSION),*.so)
	$(call prepare_tflite_runtime_debian,$(TFLITE_RUNTIME_VERSION))
	(cd $(TFLITE_RUNTIME_DIR) && \
	 export PACKAGE_VERSION=$(TFLITE_RUNTIME_VERSION) && \
	 dpkg-buildpackage -rfakeroot -us -uc -tc -b &&  \
	 dpkg-buildpackage -rfakeroot -us -uc -tc -b -a armhf -d && \
	 dpkg-buildpackage -rfakeroot -us -uc -tc -b -a arm64 -d)
	mkdir -p $(MAKEFILE_DIR)/dist
	mv $(TFLITE_RUNTIME_DIR)/../*.{deb,changes,buildinfo} \
	   $(MAKEFILE_DIR)/dist

runtime:
	make LIBEDGETPU_BIN=$(MAKEFILE_DIR)/libedgetpu_bin DIST_DIR=$(MAKEFILE_DIR)/dist -C $(MAKEFILE_DIR)/libedgetpu runtime

help:
	@echo "make all          - Build all native code"
	@echo "make pybind       - Build pycoral native code"
	@echo "make tflite       - Build tflite_runtime native code"
	@echo "make clean        - Remove generated files"
	@echo "make deb          - Build pycoral deb packages for all platforms"
	@echo "make wheel        - Build pycoral wheel for current platform"
	@echo "make wheel-all    - Build pycoral wheel for all platforms"
	@echo "make tflite-wheel - Build tflite_runtime wheel for current platform"
	@echo "make tflite-deb   - Build tflite_runtime deb packages for all platforms"
	@echo "make runtime      - Build runtime archive"
	@echo "make help         - Print help message"

TEST_ENV := $(shell test -L $(MAKEFILE_DIR)/test_data && echo 1)
DOCKER_WORKSPACE := $(MAKEFILE_DIR)/$(if $(TEST_ENV),..,)
DOCKER_WORKSPACE_CD := $(if $(TEST_ENV),pycoral,)
DOCKER_CPUS := k8 armv7a aarch64
DOCKER_TAG_BASE := coral-edgetpu
include $(MAKEFILE_DIR)/libcoral/docker/docker.mk
