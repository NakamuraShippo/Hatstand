@echo off
chcp 65001 >nul
setlocal

pushd "%~dp0.."
call "%CD%\boot.bat"
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
