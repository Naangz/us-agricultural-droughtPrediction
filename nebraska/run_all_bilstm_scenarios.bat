@echo off
setlocal
cd /d "%~dp0"

echo Activating conda environment: env_ta
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" env_ta
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat" env_ta
) else (
    echo Could not find Anaconda/Miniconda activate.bat.
    echo Make sure conda is available on PATH or edit this file with your install path.
    pause
    exit /b 1
)

echo Running all Nebraska BiLSTM scenario scripts...
python "%~dp0run_all_bilstm_scenarios.py"
set EXIT_CODE=%errorlevel%

echo.
echo Finished with exit code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
