@echo off
rem ---------------------------------------------------------------------------
rem  example_pack_iq3.bat - pack the prometheusAIR DeepSeek V4.1 Flash IQ3_XXS
rem  quant for the Maya DS4 engine (Windows).
rem
rem  Edit MODEL_DIR / ENGRAM_DIR / OUT_DIR first.  Nothing here is machine-
rem  specific.  Disk requirement: source ~417 GB + output ~352 GB (peak ~770 GB).
rem
rem  The packer writes only pack metadata (ds4.txt, tensors.txt, engram.bin,
rem  tokenizer/) beside the weights - no weight data is copied.
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0\.."

rem --- EDIT THESE -----------------------------------------------------------
set MODEL_DIR=D:\models\DeepSeek-V4.1-Flash-1017-IQ3_XXS
set ENGRAM_DIR=%MODEL_DIR%\Engram
set OUT_DIR=%MODEL_DIR%\pack
rem Optional DSpark drafter description (JSON) for the same checkpoint:
set DSPARK_JSON=

rem --- PACK -----------------------------------------------------------------
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

set ARGS=--model "%MODEL_DIR%" --out "%OUT_DIR%" --engram "%ENGRAM_DIR%"
if defined DSPARK_JSON set ARGS=%ARGS% --drafter "%DSPARK_JSON%"

python tools\pack_gguf_patched.py %ARGS%
if errorlevel 1 ( echo PACK FAILED & exit /b 1 )

rem --- VERIFY the geometry the engine will read ------------------------------
python tools\verify_kv_mapping.py "%OUT_DIR%"
if errorlevel 1 ( echo VERIFY FAILED & exit /b 1 )

echo.
echo PACK OK: geometry verified.  Point the engine at %OUT_DIR%.
endlocal
