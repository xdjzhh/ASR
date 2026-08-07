@echo off
chcp 65001 >nul
echo ============================================================
echo  🎤 VocoType ASR 本地语音转文字服务
echo ============================================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

REM 检查依赖是否安装
echo 正在检查依赖...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo 首次运行，正在安装依赖...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo 依赖安装失败，请检查网络连接或手动运行: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo.
echo 启动服务...
echo 服务地址: http://localhost:8000
echo API文档: http://localhost:8000/docs
echo.
echo 按 Ctrl+C 停止服务
echo ============================================================
echo.

python server.py %*

pause
