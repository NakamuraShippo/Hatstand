@echo off
chcp 65001 >nul
setlocal

pushd "%~dp0"
echo Hatstand 起動スクリプト

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
set "APP_ENTRY=%CD%\app\main.py"

if not exist "%PYTHON_EXE%" (
    echo .venv\Scripts\python.exe が見つかりません
    echo 先にこのフォルダで仮想環境を作成してください
    pause
    popd
    exit /b 1
)

if not exist "%APP_ENTRY%" (
    echo app\main.py が見つかりません
    pause
    popd
    exit /b 1
)

"%PYTHON_EXE%" "%APP_ENTRY%"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo アプリケーションはエラー終了しました ^(exit code: %EXIT_CODE%^)
    pause
)

popd
exit /b %EXIT_CODE%
