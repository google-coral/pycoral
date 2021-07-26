:: Copyright 2019-2021 Google LLC
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::     https://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.

echo off
setlocal enabledelayedexpansion

if not defined PYTHON ( set PYTHON=python )

set BAZEL_CMD=bazel
if defined BAZEL_OUTPUT_BASE (
    set BAZEL_CMD=%BAZEL_CMD% --output_base=%BAZEL_OUTPUT_BASE%
)

set BAZEL_VS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools
set BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC
call "%BAZEL_VC%\Auxiliary\Build\vcvars64.bat"

for /f %%i in ('%BAZEL_CMD% info output_base') do set "BAZEL_OUTPUT_BASE=%%i"
for /f %%i in ('%BAZEL_CMD% info output_path') do set "BAZEL_OUTPUT_PATH=%%i"
for /f %%i in ('%PYTHON% -c "import sys;print(str(sys.version_info.major)+str(sys.version_info.minor))"') do set "PY3_VER=%%i"
for /f %%i in ('%PYTHON% -c "import sys;print(sys.executable)"') do set "PYTHON_BIN_PATH=%%i"
for /f %%i in ('%PYTHON% -c "import sys;print(sys.base_prefix)"') do set "PYTHON_LIB_PATH=%%i\Lib"

set BAZEL_OUTPUT_PATH=%BAZEL_OUTPUT_PATH:/=\%
set BAZEL_OUTPUT_BASE=%BAZEL_OUTPUT_BASE:/=\%
set CPU=x64_windows
set COMPILATION_MODE=opt
set LIBEDGETPU_VERSION=direct

set ROOTDIR=%~dp0\..\..\
set BAZEL_OUT_DIR=%BAZEL_OUTPUT_PATH%\%CPU%-%COMPILATION_MODE%\bin
set PYBIND_OUT_DIR=%ROOTDIR%\pycoral\pybind
set TFLITE_WRAPPER_OUT_DIR=%ROOTDIR%\tflite_runtime
set LIBEDGETPU_DIR=%ROOTDIR%\libedgetpu_bin\%LIBEDGETPU_VERSION%\x64_windows

set TFLITE_WRAPPER_NAME=_pywrap_tensorflow_interpreter_wrapper.cp%PY3_VER%-win_amd64.pyd
set PYBIND_WRAPPER_NAME=_pywrap_coral.cp%PY3_VER%-win_amd64.pyd
set LIBEDGETPU_DLL_NAME=edgetpu.dll

set TFLITE_WRAPPER_PATH=%TFLITE_WRAPPER_OUT_DIR%\%TFLITE_WRAPPER_NAME%
set PYBIND_WRAPPER_PATH=%PYBIND_OUT_DIR%\%PYBIND_WRAPPER_NAME%
set LIBEDGETPU_DLL_PATH=%LIBEDGETPU_DIR%\%LIBEDGETPU_DLL_NAME%

:PROCESSARGS
set ARG=%1
if defined ARG (
    if "%ARG%"=="/DBG" (
        set COMPILATION_MODE=dbg
    )
    shift
    goto PROCESSARGS
)

for /f "tokens=1" %%i in ('bazel query "@libedgetpu_properties//..." ^| findstr /C:"tensorflow_commit" ^| cut -d# -f2') do set "TENSORFLOW_COMMIT=%%i"
set BAZEL_BUILD_FLAGS= ^
--compilation_mode=%COMPILATION_MODE% ^
--copt=/DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION ^
--copt=/D_HAS_DEPRECATED_RESULT_OF ^
--linkopt=/DEFAULTLIB:%LIBEDGETPU_DLL_PATH%.if.lib ^
--copt=-D__PRETTY_FUNCTION__=__FUNCSIG__

rem PYBIND
%BAZEL_CMD% build %BAZEL_BUILD_FLAGS% ^
    //src:edgetpu.res || goto :exit
%BAZEL_CMD% build %BAZEL_BUILD_FLAGS% ^
    --embed_label=%TENSORFLOW_COMMIT% ^
    --stamp ^
    //src:_pywrap_coral || goto :exit
if not exist %PYBIND_OUT_DIR% md %PYBIND_OUT_DIR%
type NUL >%PYBIND_OUT_DIR%\__init__.py
copy %BAZEL_OUT_DIR%\src\_pywrap_coral.pyd %PYBIND_WRAPPER_PATH% >NUL

rem TfLite
%BAZEL_CMD% build %BAZEL_BUILD_FLAGS% ^
    @org_tensorflow//tensorflow/lite/python/interpreter_wrapper:_pywrap_tensorflow_interpreter_wrapper || goto :exit
if not exist %TFLITE_WRAPPER_OUT_DIR% md %TFLITE_WRAPPER_OUT_DIR%
copy %BAZEL_OUT_DIR%\external\org_tensorflow\tensorflow\lite\python\interpreter_wrapper\_pywrap_tensorflow_interpreter_wrapper.pyd ^
     %TFLITE_WRAPPER_PATH% >NUL
copy %BAZEL_OUTPUT_BASE%\external\org_tensorflow\tensorflow\lite\python\interpreter.py %TFLITE_WRAPPER_OUT_DIR%
copy %BAZEL_OUTPUT_BASE%\external\org_tensorflow\tensorflow\lite\python\metrics_interface.py %TFLITE_WRAPPER_OUT_DIR%
copy %BAZEL_OUTPUT_BASE%\external\org_tensorflow\tensorflow\lite\python\metrics_portable.py %TFLITE_WRAPPER_OUT_DIR%
type NUL >%TFLITE_WRAPPER_OUT_DIR%\__init__.py

:exit
exit /b %ERRORLEVEL%
