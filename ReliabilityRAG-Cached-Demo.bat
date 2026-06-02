@echo off
REM ===================================================================
REM  ReliabilityRAG Cached Demo (GPU yok / yedek mod)
REM  GPU sorunu varsa veya hizli onizleme icin: bu dosyayi cift tikla.
REM  Onceden hesaplanmis demo cache'ten cevap gosterir.
REM ===================================================================

title ReliabilityRAG (Cached)

cd /d "%~dp0"
call conda activate base >nul 2>&1

REM Tarayiciyi 5 saniye sonra ac (cached mode hizli baslar)
start "" /b cmd /c "timeout /t 5 /nobreak >nul && start http://127.0.0.1:7861"

echo.
echo ========================================================
echo  ReliabilityRAG (Cached Demo) baslatiliyor...
echo  Tarayici ~5 saniye sonra acilacak.
echo  Manuel: http://127.0.0.1:7861
echo ========================================================
echo.

python cached_demo.py

echo.
pause
