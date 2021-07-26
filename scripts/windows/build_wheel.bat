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
set ROOTDIR=%~dp0\..\..\
for /f %%i in ('%PYTHON% -c "import sys;print(str(sys.version_info.major)+str(sys.version_info.minor))"') do set "PY3_VER=%%i"
set WRAPPER_NAME=_pywrap_coral.cp%PY3_VER%-win_amd64.pyd

rem Build the code, in case it doesn't exist yet.
call %ROOTDIR%\scripts\windows\build.bat || goto :exit

%PYTHON% %ROOTDIR%\setup.py bdist_wheel -d %ROOTDIR%\dist
rd /s /q build

:exit
exit /b %ERRORLEVEL%