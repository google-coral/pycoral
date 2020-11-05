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