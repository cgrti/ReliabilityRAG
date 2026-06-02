@echo off
REM ===================================================================
REM  ReliabilityRAG Cached Demo (GPU yok / yedek mod)
REM  GPU sorunu varsa veya hizli onizleme icin: cift tikla.
REM ===================================================================

title ReliabilityRAG (Cached)
cd /d "%~dp0"

REM ---- Conda 'base' aktivasyonu ----
set CONDA_ACTIVATED=0
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" base
    set CONDA_ACTIVATED=1
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat" base
    set CONDA_ACTIVATED=1
) else if exist "C:\ProgramData\Anaconda3\Scripts\activate.bat" (
    call "C:\ProgramData\Anaconda3\Scripts\activate.bat" base
    set CONDA_ACTIVATED=1
) else if exist "C:\ProgramData\Miniconda3\Scripts\activate.bat" (
    call "C:\ProgramData\Miniconda3\Scripts\activate.bat" base
    set CONDA_ACTIVATED=1
) else if exist "C:\Anaconda3\Scripts\activate.bat" (
    call "C:\Anaconda3\Scripts\activate.bat" base
    set CONDA_ACTIVATED=1
)

where python >nul 2>&1
if errorlevel 1 (
    echo [HATA] python bulunamadi.
    pause
    exit /b 1
)

set PYTHONIOENCODING=utf-8

REM Cached mod 5s'de hazir
start "" /b cmd /c "timeout /t 5 /nobreak >nul && start http://127.0.0.1:7861"

echo.
echo ========================================================
echo  ReliabilityRAG (Cached Demo) baslatiliyor...
echo  Tarayici ~5s sonra acilacak.
echo  Manuel: http://127.0.0.1:7861
echo ========================================================
echo.

python cached_demo.py

echo.
echo ========================================================
echo  Demo durdu. Pencereyi kapatabilirsiniz.
echo ========================================================
pause
