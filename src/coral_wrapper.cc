/* Copyright 2019-2021 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <Python.h>
#include <numpy/arrayobject.h>

#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "coral/bbox.h"
#include "coral/learn/backprop/softmax_regression_model.h"
#include "coral/learn/imprinting_engine.h"
#include "coral/learn/utils.h"
#include "coral/pipeline/allocator.h"
#include "coral/pipeline/common.h"
#include "coral/pipeline/pipelined_model_runner.h"
#include "coral/tflite_utils.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/stateful_error_reporter.h"
#include "tflite/public/edgetpu.h"

namespace {
namespace py = pybind11;

template <size_t SizeOf, bool IsSigned>
struct NumPyTypeImpl;

template <>
struct NumPyTypeImpl<4, true> {
  enum { type = NPY_INT32 };
};

template <>
struct NumPyTypeImpl<4, false> {
  enum { type = NPY_UINT32 };
};

template <>
struct NumPyTypeImpl<8, true> {
  enum { type = NPY_INT64 };
};

template <>
struct NumPyTypeImpl<8, false> {
  enum { type = NPY_UINT64 };
};

template <typename T>
struct NumPyType {
  enum { type = NumPyTypeImpl<sizeof(T), std::is_signed<T>::value>::type };
};

template <typename T>
PyObject* PyArrayFromSpan(absl::Span<T> span) {
  npy_intp size = span.size();
  void* pydata = malloc(size * sizeof(T));
  std::memcpy(pydata, span.data(), size * sizeof(T));

  PyObject* obj = PyArray_SimpleNewFromData(
      1, &size, NumPyType<typename std::remove_cv<T>::type>::type, pydata);
  PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(obj), NPY_ARRAY_OWNDATA);
  return obj;
}

py::object Pyo(PyObject* ptr) { return py::reinterpret_steal<py::object>(ptr); }

using Strides = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
using Scalar = Eigen::MatrixXf::Scalar;
constexpr bool kRowMajor = Eigen::MatrixXf::Flags & Eigen::RowMajorBit;

Eigen::MatrixXf TensorFromPyBuf(const py::buffer& b) {
  py::buffer_info info = b.request();
  if (info.format != py::format_descriptor<Scalar>::format())
    throw std::runtime_error("Incompatible format: expected a float array!");
  if (info.ndim != 2)
    throw std::runtime_error("Incompatible buffer dimension!");
  auto strides = Strides(info.strides[kRowMajor ? 0 : 1] / sizeof(Scalar),
                         info.strides[kRowMajor ? 1 : 0] / sizeof(Scalar));
  auto map = Eigen::Map<Eigen::MatrixXf, 0, Strides>(
      static_cast<Scalar*>(info.ptr), info.shape[0], info.shape[1], strides);
  return Eigen::MatrixXf(map);
}

template <typename T>
absl::Span<T> BufferInfoSpan(const py::buffer_info& info) {
  return absl::MakeSpan(reinterpret_cast<T*>(info.ptr), info.size);
}

std::unique_ptr<tflite::FlatBufferModel> LoadModel(
    const std::string& model_path) {
  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  if (!model) throw std::invalid_argument("Failed to open file: " + model_path);
  return model;
}

template <typename T>
py::bytes SerializeModel(T& engine) {
  flatbuffers::FlatBufferBuilder fbb;
  auto status = engine.SerializeModel(&fbb);
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
  return py::bytes(reinterpret_cast<char*>(fbb.GetBufferPointer()),
                   fbb.GetSize());
}

std::string GetRuntimeVersion() {
  return ::edgetpu::EdgeTpuManager::GetSingleton()->Version();
}

TfLiteType NumpyDtypeToTfLiteType(const std::string& format) {
  static std::unordered_map<std::string, TfLiteType>* type_map =
      new std::unordered_map<std::string, TfLiteType>{
          {py::format_descriptor<float>::format(), kTfLiteFloat32},
          {py::format_descriptor<int32_t>::format(), kTfLiteInt32},
          {py::format_descriptor<uint8_t>::format(), kTfLiteUInt8},
          {py::format_descriptor<int64_t>::format(), kTfLiteInt64},
          {py::format_descriptor<int16_t>::format(), kTfLiteInt16},
          {py::format_descriptor<int8_t>::format(), kTfLiteInt8},
          {py::format_descriptor<double>::format(), kTfLiteFloat64},
      };
  const auto it = type_map->find(format);
  if (it == type_map->end()) {
    throw std::runtime_error("Unexpected numpy dtype: " + format);
  } else {
    return it->second;
  }
}

py::dtype TfLiteTypeToNumpyDtype(const TfLiteType& type) {
  // std::hash<int> is added here because of a defect in std::unordered_map API,
  // which is fixed in C++14 and newer version of libstdc++.
  // https://stackoverflow.com/a/29618545
  static std::unordered_map<TfLiteType, std::string, std::hash<int>>* type_map =
      new std::unordered_map<TfLiteType, std::string, std::hash<int>>{
          {kTfLiteFloat32, py::format_descriptor<float>::format()},
          {kTfLiteInt32, py::format_descriptor<int32_t>::format()},
          {kTfLiteUInt8, py::format_descriptor<uint8_t>::format()},
          {kTfLiteInt64, py::format_descriptor<int64_t>::format()},
          {kTfLiteInt16, py::format_descriptor<int16_t>::format()},
          {kTfLiteInt8, py::format_descriptor<int8_t>::format()},
          {kTfLiteFloat64, py::format_descriptor<double>::format()},
      };
  const auto it = type_map->find(type);
  if (it == type_map->end()) {
    throw std::runtime_error("Unexpected TfLiteType: " +
                             std::string(TfLiteTypeGetName(type)));
  } else {
    return py::dtype(it->second);
  }
}

class MallocBuffer : public coral::Buffer {
 public:
  explicit MallocBuffer(void* ptr) : ptr_(ptr) {}

  void* ptr() override { return ptr_; }

 private:
  void* ptr_ = nullptr;
};

// Allocator with leaky `free` function. Caller should use std::free() to free
// the underlying memory allocated by std::malloc; otherwise there will be
// memory leaks.
class LeakyMallocAllocator : public coral::Allocator {
 public:
  LeakyMallocAllocator() = default;

  coral::Buffer* Alloc(size_t size) override {
    return new MallocBuffer(std::malloc(size));
  }

  void Free(coral::Buffer* buffer) override {
    // Note: the memory allocated by std::malloc is not freed here.
    delete buffer;
  }
};

}  // namespace

PYBIND11_MODULE(_pywrap_coral, m) {
  // This function must be called in the initialization section of a module that
  // will make use of the C-API (PyArray_SimpleNewFromData).
  // It imports the module where the function-pointer table is stored and points
  // the correct variable to it.
  // Different with import_array() import_array1() has return value.
  // https://docs.scipy.org/doc/numpy-1.14.2/reference/c-api.array.html
  import_array1();
  py::options options;
  options.disable_function_signatures();

  m.def(
      "InvokeWithMemBuffer",
      [](py::object interpreter_handle, uintptr_t buffer, size_t size) {
        auto* interpreter = reinterpret_cast<tflite::Interpreter*>(
            interpreter_handle.cast<intptr_t>());
        py::gil_scoped_release release;
        auto status = coral::InvokeWithMemBuffer(
            interpreter, reinterpret_cast<void*>(buffer), size,
            static_cast<tflite::StatefulErrorReporter*>(
                interpreter->error_reporter()));
        if (!status.ok())
          throw std::runtime_error(std::string(status.message()));
      },
      R"pbdoc(
        Invoke the given ``tf.lite.Interpreter`` with a pointer to a native
        memory allocation.

        Works only for Edge TPU models running on PCIe TPU devices.

        Args:
          interpreter: The ``tf.lite:Interpreter`` to invoke.
          buffer (intptr_t): Pointer to memory buffer with input data.
          size (size_t): The buffer size.
      )pbdoc");

  m.def(
      "InvokeWithBytes",
      [](py::object interpreter_handle, py::bytes input_data) {
        auto* interpreter = reinterpret_cast<tflite::Interpreter*>(
            interpreter_handle.cast<intptr_t>());
        char* buffer;
        ssize_t length;
        PyBytes_AsStringAndSize(input_data.ptr(), &buffer, &length);
        py::gil_scoped_release release;
        auto status = coral::InvokeWithMemBuffer(
            interpreter, buffer, static_cast<size_t>(length),
            static_cast<tflite::StatefulErrorReporter*>(
                interpreter->error_reporter()));
        if (!status.ok())
          throw std::runtime_error(std::string(status.message()));
      },
      R"pbdoc(
        Invoke the given ``tf.lite.Interpreter`` with bytes as input.

        Args:
          interpreter: The ``tf.lite:Interpreter`` to invoke.
          input_data (bytes): Raw bytes as input data.
      )pbdoc");

  m.def(
      "InvokeWithDmaBuffer",
      [](py::object interpreter_handle, int dma_fd, size_t size) {
        auto* interpreter = reinterpret_cast<tflite::Interpreter*>(
            interpreter_handle.cast<intptr_t>());
        py::gil_scoped_release release;
        auto status = coral::InvokeWithDmaBuffer(
            interpreter, dma_fd, size,
            static_cast<tflite::StatefulErrorReporter*>(
                interpreter->error_reporter()));
        if (!status.ok())
          throw std::runtime_error(std::string(status.message()));
      },
      R"pbdoc(
        Invoke the given ``tf.lite.Interpreter`` using a given Linux dma-buf
        file descriptor as an input tensor.

        Works only for Edge TPU models running on PCIe-based Coral devices.
        You can verify device support with ``supports_dmabuf()``.

        Args:
          interpreter: The ``tf.lite:Interpreter`` to invoke.
          dma_fd (int): DMA file descriptor.
          size (size_t): DMA buffer size.
      )pbdoc");

  m.def(
      "SupportsDmabuf",
      [](py::object interpreter_handle) {
        auto* interpreter = reinterpret_cast<tflite::Interpreter*>(
            interpreter_handle.cast<intptr_t>());
        auto* context = interpreter->primary_subgraph().context();
        auto* edgetpu_context = static_cast<edgetpu::EdgeTpuContext*>(
            context->GetExternalContext(context, kTfLiteEdgeTpuContext));
        if (!edgetpu_context) return false;
        auto device = edgetpu_context->GetDeviceEnumRecord();
        return device.type == edgetpu::DeviceType::kApexPci;
      },
      R"pbdoc(
        Checks whether the device supports Linux dma-buf.

        Args:
          interpreter: The ``tf.lite:Interpreter`` that's bound to the
            Edge TPU you want to query.
        Returns:
          True if the device supports DMA buffers.
      )pbdoc");

  m.def("GetRuntimeVersion", &GetRuntimeVersion,
        R"pbdoc(
        Returns the Edge TPU runtime (libedgetpu.so) version.

        This runtime version is dynamically retrieved from the shared object.

        Returns:
          A string for the version name.
      )pbdoc");

  m.def(
      "ListEdgeTpus",
      []() {
        py::list device_list;
        for (const auto& item :
             edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu()) {
          py::dict device;
          device["type"] =
              item.type == edgetpu::DeviceType::kApexPci ? "pci" : "usb";
          device["path"] = item.path;
          device_list.append(device);
        }
        return device_list;
      },
      R"pbdoc(
      Lists all available Edge TPU devices.

      Returns:
        A list of dictionary items, each representing an Edge TPU in the system.
        Each dictionary includes a "type" (either "usb" or "pci") and a
        "path" (the device location in the system). Note: The order of the
        Edge TPUs in this list are not guaranteed to be consistent across
        system reboots.
    )pbdoc");

  m.def(
      "SetVerbosity",
      [](int verbosity) {
        auto status =
            edgetpu::EdgeTpuManager::GetSingleton()->SetVerbosity(verbosity);
        return status == TfLiteStatus::kTfLiteOk;
      },
      R"pbdoc(
      Sets the verbosity of operating logs related to each Edge TPU.
      10 is the most verbose; 0 is the default.

      Args:
        verbosity(int): Desired verbosity 0-10.
      Returns:
        A boolean indicating if verbosity was succesfully set.
    )pbdoc");

  py::class_<coral::ImprintingEngine>(m, "ImprintingEnginePythonWrapper")
      .def(py::init([](const std::string& model_path, bool keep_classes) {
        std::unique_ptr<coral::ImprintingModel> model;
        auto status = coral::ImprintingModel::Create(
            *LoadModel(model_path)->GetModel(), &model);
        if (!status.ok())
          throw std::invalid_argument(std::string(status.message()));
        return coral::ImprintingEngine::Create(std::move(model), keep_classes);
      }))
      .def("EmbeddingDim",
           [](coral::ImprintingEngine& self) { return self.embedding_dim(); })
      .def("NumClasses",
           [](coral::ImprintingEngine& self) {
             return self.GetClasses().size();
           })
      .def("SerializeExtractorModel",
           [](coral::ImprintingEngine& self) {
             auto buffer = self.ExtractorModelBuffer();
             return py::bytes(buffer.data(), buffer.size());
           })
      .def("SerializeModel",
           [](coral::ImprintingEngine& self) { return SerializeModel(self); })
      .def("Train", [](coral::ImprintingEngine& self,
                       py::array_t<float> weights_array, int class_id) {
        auto request = weights_array.request();
        if (request.shape != std::vector<ssize_t>{self.embedding_dim()})
          throw std::runtime_error("Invalid weights array shape.");

        const auto* weights = reinterpret_cast<float*>(request.ptr);
        auto status =
            self.Train(absl::MakeSpan(weights, self.embedding_dim()), class_id);
        if (!status.ok())
          throw std::runtime_error(std::string(status.message()));
      });
  py::class_<coral::TrainConfig>(m, "TrainConfigWrapper")
      .def(py::init<int, int, int>());
  py::class_<coral::TrainingData>(m, "TrainingDataWrapper")
      .def(py::init<>([](const py::buffer& training_data,
                         const py::buffer& validation_data,
                         const std::vector<int>& training_labels,
                         const std::vector<int>& validation_labels) {
        auto self = absl::make_unique<coral::TrainingData>();
        self->training_data = TensorFromPyBuf(training_data);
        self->validation_data = TensorFromPyBuf(validation_data);
        self->training_labels = training_labels;
        self->validation_labels = validation_labels;
        return self;
      }));
  py::class_<coral::SoftmaxRegressionModel>(m, "SoftmaxRegressionModelWrapper")
      .def(py::init<int, int, float, float>())
      .def("Train",
           [](coral::SoftmaxRegressionModel& self,
              const coral::TrainingData& training_data,
              const coral::TrainConfig& train_config, float learning_rate) {
             return self.Train(training_data, train_config, learning_rate);
           })
      .def("GetAccuracy",
           [](coral::SoftmaxRegressionModel& self,
              const py::buffer& training_data,
              const std::vector<int>& training_labels) {
             return self.GetAccuracy(TensorFromPyBuf(training_data),
                                     training_labels);
           })
      .def("AppendLayersToEmbeddingExtractor",
           [](coral::SoftmaxRegressionModel& self,
              const std::string& in_model_path) {
             flatbuffers::FlatBufferBuilder fbb;
             self.AppendLayersToEmbeddingExtractor(
                 *LoadModel(in_model_path)->GetModel(), &fbb);
             return py::bytes(reinterpret_cast<char*>(fbb.GetBufferPointer()),
                              fbb.GetSize());
           });

  py::class_<coral::PipelinedModelRunner>(m, "PipelinedModelRunnerWrapper")
      .def(py::init([](const py::list& list) {
        static coral::Allocator* output_tensor_allocator =
            new LeakyMallocAllocator();
        std::vector<tflite::Interpreter*> interpreters(list.size());
        for (int i = 0; i < list.size(); ++i) {
          interpreters[i] =
              reinterpret_cast<tflite::Interpreter*>(list[i].cast<intptr_t>());
        }
        return absl::make_unique<coral::PipelinedModelRunner>(
            interpreters, /*input_tensor_allocator=*/nullptr,
            output_tensor_allocator);
      }))
      .def("SetInputQueueSize", &coral::PipelinedModelRunner::SetInputQueueSize)
      .def("SetOutputQueueSize",
           &coral::PipelinedModelRunner::SetOutputQueueSize)
      .def("Push",
           [](coral::PipelinedModelRunner& self, py::dict& input_tensor_dict) {
             std::vector<coral::PipelineTensor> input_tensors(
                 input_tensor_dict.size());
             int i = 0;
             for (const auto& item : input_tensor_dict) {
               input_tensors[i].name = item.first.cast<std::string>();
               const auto info = item.second.cast<py::buffer>().request();
               input_tensors[i].type = NumpyDtypeToTfLiteType(info.format);
               input_tensors[i].bytes = info.size * info.itemsize;
               input_tensors[i].buffer = self.GetInputTensorAllocator()->Alloc(
                   input_tensors[i].bytes);
               std::memcpy(input_tensors[i].buffer->ptr(), info.ptr,
                           input_tensors[i].bytes);
               ++i;
             }
             // Release GIL because Push can be blocking (if input queue size is
             // bigger than input queue size threshold).
             py::gil_scoped_release release;
             const auto push_status = self.Push(input_tensors);
             py::gil_scoped_acquire acquire;
             if (!push_status.ok()) {
               throw std::runtime_error(std::string(push_status.message()));
             }
           })
      .def("Pop", [](coral::PipelinedModelRunner& self) -> py::object {
        std::vector<coral::PipelineTensor> output_tensors;

        // Release GIL because Pop is blocking.
        py::gil_scoped_release release;
        const auto pop_status = self.Pop(&output_tensors);
        py::gil_scoped_acquire acquire;

        if (!pop_status.ok()) {
          throw std::runtime_error(std::string(pop_status.message()));
        }

        if (output_tensors.empty()) {
          return py::none();
        }

        py::dict result;
        for (auto tensor : output_tensors) {
          // Underlying memory's ownership is passed to numpy object.
          py::capsule free_when_done(tensor.buffer->ptr(),
                                     [](void* ptr) { std::free(ptr); });
          result[PyUnicode_DecodeLatin1(tensor.name.data(), tensor.name.size(),
                                        nullptr)] =
              py::array(TfLiteTypeToNumpyDtype(tensor.type),
                        /*shape=*/{tensor.bytes},
                        /*strides=*/{1}, tensor.buffer->ptr(), free_when_done);
          self.GetOutputTensorAllocator()->Free(tensor.buffer);
        }
        return result;
      });
}
