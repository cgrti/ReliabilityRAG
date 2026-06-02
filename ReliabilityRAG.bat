@echo off
REM ===================================================================
REM  ReliabilityRAG — Tek tik launcher
REM  Cift tikla -> Gradio uygulamasi baslar, tarayicida acilir.
REM ===================================================================

title ReliabilityRAG
cd /d "%~dp0"

REM ---- Conda 'base' ortamini robust sekilde aktive et --------------
REM 'conda activate' calismaz cunku conda init shell hook'lari double-click
REM acilan cmd'de yuklenmez. Bunun yerine activate.bat'i direkt cagiririz,
REM standart Windows kurulum yollarini deneriz.

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

REM ---- python.exe'yi bul ------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo [HATA] python bulunamadi.
    echo Conda kurulu degil veya PATH'te yok.
    echo.
    pause
    exit /b 1
)

REM ---- Env vars ---------------------------------------------------
set HF_HUB_ENABLE_HF_TRANSFER=1
set PYTHONIOENCODING=utf-8

REM ---- Tarayiciyi 25 saniye sonra otomatik ac ---------------------
start "" /b cmd /c "timeout /t 25 /nobreak >nul && start http://127.0.0.1:7860"

echo.
echo ========================================================
echo  ReliabilityRAG baslatiliyor...
if "%CONDA_ACTIVATED%"=="1" (
    echo  Conda base ortami: AKTIF
) else (
    echo  Conda bulunamadi - sistem python kullaniliyor
)
echo  Tarayici ~25s sonra otomatik acilacak.
echo  Manuel: http://127.0.0.1:7860
echo  Kapatmak icin bu pencereyi kapatin.
echo ========================================================
echo.

python app.py

echo.
echo ========================================================
echo  Uygulama durdu. Pencereyi kapatabilirsiniz.
echo ========================================================
pause
