@echo off
rem 部署 qgis_yolo_annotator 插件到当前 QGIS profile 并检查依赖
setlocal
set PLUGIN_NAME=qgis_yolo_annotator
set SRC=%~dp0..\src\%PLUGIN_NAME%
set DST=%APPDATA%\QGIS\QGIS4\profiles\default\python\plugins\%PLUGIN_NAME%
set PYQGIS=D:\Tools\QGIS\bin\python-qgis.bat

if not exist "%SRC%\metadata.txt" (
    echo [ERROR] source not found: %SRC%
    exit /b 1
)

echo [1/3] deploy %SRC% -^> %DST%
if exist "%DST%" rmdir /s /q "%DST%"
mkdir "%DST%"
xcopy /e /i /y "%SRC%" "%DST%" >nul
if errorlevel 1 (
    echo [ERROR] copy failed
    exit /b 1
)

echo [2/3] check python deps in QGIS python
cmd /c "%PYQGIS% -c "import onnxruntime, cv2; print('deps ok: onnxruntime', onnxruntime.__version__, 'cv2', cv2.__version__)" 2>nul
if errorlevel 1 (
    echo [WARN] onnxruntime/cv2 missing, installing...
    cmd /c "%PYQGIS% -m pip install -r %~dp0..\requirements-qgis.txt"
)

echo [3/3] done. restart QGIS or use Plugin Manager to reload.
endlocal
