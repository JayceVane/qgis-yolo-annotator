@echo off
rem 打包插件 zip（调 scripts/package.py，产物 dist/qgis_yolo_annotator-<ver>.zip）
setlocal
set HERE=%~dp0..
for /f "tokens=*" %%p in ('dir /b /s "D:\Tools\QGIS\bin\python-qgis.bat" 2^>nul') do set QGIS_PY=%%p
if "%QGIS_PY%"=="" set QGIS_PY=D:\Tools\QGIS\bin\python-qgis.bat
call "%QGIS_PY%" "%~dp0package.py"
endlocal
