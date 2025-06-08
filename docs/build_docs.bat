@echo off
REM Script to build the documentation on Windows

:: Set Python executable (use python3 if available, otherwise python)
set PYTHON=python

:: Check if python3 is available
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    python3 --version >nul 2>&1
    if %ERRORLEVEL% EQU 0 set PYTHON=python3
)

:: Generate API documentation
echo Generating API documentation...
%PYTHON% generate_api_docs.py

:: Create _static directory if it doesn't exist
if not exist _static mkdir _static

:: Build the documentation
echo Building documentation...
%PYTHON% -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Documentation built successfully!
    echo Open _build\html\index.html in your browser to view the documentation.
) else (
    echo.
    echo Error building documentation. Check the output above for details.
    exit /b 1
)
