@echo off
rem ---------------------------------------------------------------------------
rem  example_pack_iq2.bat - pack the Abiray DeepSeek V4.1 Flash IQ2_XXS
rem  (uncensored GSQ-RCO) quant for the Maya DS4 engine (Windows).
rem
rem  The 6-shard layout shares one Engram file (shard 6).  Download it with
rem  tools\download_hf_model.py first, then hardlink shard 6 to the Engram
rem  file (see README - "Install").
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0\.."

rem --- EDIT THESE -----------------------------------------------------------
set MODEL_DIR=D:\models\DeepSeek-V4.1-Uncensored-IQ2XXS
set ENGRAM_FILE=%MODEL_DIR%\DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf
set OUT_DIR=%MODEL_DIR%\pack

rem --- LINK shard 6 to the Engram file (idempotent) --------------------------
if exist "%ENGRAM_FILE%" if not exist "%MODEL_DIR%\DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00006-of-00006.gguf" (
    mklink /H "%MODEL_DIR%\DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00006-of-00006.gguf" "%ENGRAM_FILE%"
)

rem --- PACK ------------------------------------------------------------------
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
python tools\pack_gguf_patched.py --model "%MODEL_DIR%" --out "%OUT_DIR%" --engram "%MODEL_DIR%"
if errorlevel 1 ( echo PACK FAILED & exit /b 1 )

python tools\verify_kv_mapping.py "%OUT_DIR%"
if errorlevel 1 ( echo VERIFY FAILED & exit /b 1 )

echo.
echo PACK OK: geometry verified.  Point the engine at %OUT_DIR%.
endlocal
