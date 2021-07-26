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

set ROOTDIR=%~dp0\..\..\
set TEST_DATA_DIR=%ROOTDIR%\..\test_data
set LIBCORAL_DIR=%ROOTDIR%\..\libcoral
set LIBEDGETPU_DIR=%ROOTDIR%\..\libedgetpu
for /f %%i in ("%ROOTDIR%") do set "ROOTDIR=%%~fi"
for /f %%i in ("%TEST_DATA_DIR%") do set "TEST_DATA_DIR=%%~fi"
for /f %%i in ("%LIBCORAL_DIR%") do set "LIBCORAL_DIR=%%~fi"
for /f %%i in ("%LIBEDGETPU_DIR%") do set "LIBEDGETPU_DIR=%%~fi"
for /f "tokens=2 delims==" %%i in ('wmic os get /format:value ^| findstr TotalVisibleMemorySize') do set /A "MEM_KB=%%i >> 1"

rem Build Python 3.5 wheel
call %ROOTDIR%\scripts\windows\clean.bat
docker run -m %MEM_KB%KB --cpus %NUMBER_OF_PROCESSORS% --rm ^
    -v %ROOTDIR%:c:\edgetpu ^
    -v %TEST_DATA_DIR%:c:\edgetpu\test_data ^
    -v %LIBCORAL_DIR%:c:\edgetpu\libcoral ^
    -v %LIBEDGETPU_DIR%:c:\edgetpu\libedgetpu ^
    -w c:\edgetpu ^
    -e BAZEL_OUTPUT_BASE=c:\temp\edgetpu ^
    -e PYTHON=c:\python35\python.exe edgetpu-win scripts\windows\build_wheel.bat

rem Build Python 3.6 wheel
call %ROOTDIR%\scripts\windows\clean.bat
docker run -m %MEM_KB%KB --cpus %NUMBER_OF_PROCESSORS% --rm ^
    -v %ROOTDIR%:c:\edgetpu ^
    -v %TEST_DATA_DIR%:c:\edgetpu\test_data ^
    -v %LIBCORAL_DIR%:c:\edgetpu\libcoral ^
    -v %LIBEDGETPU_DIR%:c:\edgetpu\libedgetpu ^
    -w c:\edgetpu ^
    -e BAZEL_OUTPUT_BASE=c:\temp\edgetpu ^
    -e PYTHON=c:\python36\python.exe edgetpu-win scripts\windows\build_wheel.bat

rem Build Python 3.7 wheel
call %ROOTDIR%\scripts\windows\clean.bat
docker run -m %MEM_KB%KB --cpus %NUMBER_OF_PROCESSORS% --rm ^
    -v %ROOTDIR%:c:\edgetpu ^
    -v %TEST_DATA_DIR%:c:\edgetpu\test_data ^
    -v %LIBCORAL_DIR%:c:\edgetpu\libcoral ^
    -v %LIBEDGETPU_DIR%:c:\edgetpu\libedgetpu ^
    -w c:\edgetpu ^
    -e BAZEL_OUTPUT_BASE=c:\temp\edgetpu ^
    -e PYTHON=c:\python37\python.exe edgetpu-win scripts\windows\build_wheel.bat

rem Build Python 3.8 wheel
call %ROOTDIR%\scripts\windows\clean.bat
docker run -m %MEM_KB%KB --cpus %NUMBER_OF_PROCESSORS% --rm ^
    -v %ROOTDIR%:c:\edgetpu ^
    -v %TEST_DATA_DIR%:c:\edgetpu\test_data ^
    -v %LIBCORAL_DIR%:c:\edgetpu\libcoral ^
    -v %LIBEDGETPU_DIR%:c:\edgetpu\libedgetpu ^
    -w c:\edgetpu ^
    -e BAZEL_OUTPUT_BASE=c:\temp\edgetpu ^
    -e PYTHON=c:\python38\python.exe edgetpu-win scripts\windows\build_wheel.bat

rem Build Python 3.9 wheel
call %ROOTDIR%\scripts\windows\clean.bat
docker run -m %MEM_KB%KB --cpus %NUMBER_OF_PROCESSORS% --rm ^
    -v %ROOTDIR%:c:\edgetpu ^
    -v %TEST_DATA_DIR%:c:\edgetpu\test_data ^
    -v %LIBCORAL_DIR%:c:\edgetpu\libcoral ^
    -v %LIBEDGETPU_DIR%:c:\edgetpu\libedgetpu ^
    -w c:\edgetpu ^
    -e BAZEL_OUTPUT_BASE=c:\temp\edgetpu ^
    -e PYTHON=c:\python39\python.exe edgetpu-win scripts\windows\build_wheel.bat
