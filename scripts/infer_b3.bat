@echo off
REM Inference for B3 (transformer encoder) on the first 20 test respondents.
setlocal

pushd "%~dp0.." || exit /b 1
if not defined PYTHON if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
if not defined PYTHON set "PYTHON=python"

"%PYTHON%" scripts\infer.py --baseline b3 --n 20 --out recs_b3.json %*
set "EXITCODE=%ERRORLEVEL%"

popd
exit /b %EXITCODE%
