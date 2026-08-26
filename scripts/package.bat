@echo off
rem 打包插件 zip（用于发布或安装到其他机器）
setlocal
set PLUGIN_NAME=qgis_yolo_annotator
set STAGE=%TEMP%\%PLUGIN_NAME%_pkg
set OUT=%~dp0..\dist\%PLUGIN_NAME%.zip

if exist "%STAGE%" rmdir /s /q "%STAGE%"
mkdir "%STAGE%"
xcopy /e /i /y "%~dp0..\src\%PLUGIN_NAME%" "%STAGE%\%PLUGIN_NAME%" >nul
del /s /q "%STAGE%\%PLUGIN_NAME%\__pycache__" 2>nul
for /d /r "%STAGE%" %%d in (__pycache__) do rmdir /s /q "%%d" 2>nul

if not exist "%~dp0..\dist" mkdir "%~dp0..\dist"
powershell -Command "Compress-Archive -Path '%STAGE%\*' -DestinationPath '%OUT%' -Force"
echo packaged: %OUT%
endlocal
