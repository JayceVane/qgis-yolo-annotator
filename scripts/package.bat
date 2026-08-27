@echo off
rem 打包插件 zip（版本号取自 metadata.txt，产物 dist/qgis_yolo_annotator-<ver>.zip）
setlocal
set PLUGIN_NAME=qgis_yolo_annotator
set STAGE=%TEMP%\%PLUGIN_NAME%_pkg
for /f "tokens=2 delims==" %%v in ('findstr /b "version=" "%~dp0..\src\%PLUGIN_NAME%\metadata.txt"') do set VER=%%v
set OUT=%~dp0..\dist\%PLUGIN_NAME%-%VER%.zip

if exist "%STAGE%" rmdir /s /q "%STAGE%"
mkdir "%STAGE%"
xcopy /e /i /y "%~dp0..\src\%PLUGIN_NAME%" "%STAGE%\%PLUGIN_NAME%" >nul
del /s /q "%STAGE%\%PLUGIN_NAME%\__pycache__" 2>nul
for /d /r "%STAGE%" %%d in (__pycache__) do rmdir /s /q "%%d" 2>nul

if not exist "%~dp0..\dist" mkdir "%~dp0..\dist"
powershell -Command "Compress-Archive -Path '%STAGE%\*' -DestinationPath '%OUT%' -Force"
echo packaged: %OUT%
endlocal
