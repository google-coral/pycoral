echo off
setlocal enabledelayedexpansion

if not defined PY3_VER set PY3_VER=38
set ROOTDIR=%~dp0\..\..\
set TEST_DATA_DIR=%ROOTDIR%\..\test_data
set LIBCORAL_DIR=%ROOTDIR%\..\libcoral
set LIBEDGETPU_DIR=%ROOTDIR%\..\libedgetpu
for /f %%i in ("%ROOTDIR%") do set "ROOTDIR=%%~fi"
for /f %%i in ("%TEST_DATA_DIR%") do set "TEST_DATA_DIR=%%~fi"
for /f %%i in ("%LIBCORAL_DIR%") do set "LIBCORAL_DIR=%%~fi"
for /f %%i in ("%LIBEDGETPU_DIR%") do set "LIBEDGETPU_DIR=%%~fi"
for /f "tokens=2 delims==" %%i in ('wmic os get /format:value ^| findstr TotalVisibleMemorySize') do set /A "MEM_KB=%%i >> 1"

docker run -m %MEM_KB%KB --cpus %NUMBER_OF_PROCESSORS% --rm ^
    -v %ROOTDIR%:c:\edgetpu ^
    -v %TEST_DATA_DIR%:c:\edgetpu\test_data ^
    -v %LIBCORAL_DIR%:c:\edgetpu\libcoral ^
    -v %LIBEDGETPU_DIR%:c:\edgetpu\libedgetpu ^
    -w c:\edgetpu ^
    -e PYTHON=c:\python%PY3_VER%\python.exe ^
    -e BAZEL_OUTPUT_BASE=c:\temp\edgetpu ^
    edgetpu-win scripts\windows\build.bat
