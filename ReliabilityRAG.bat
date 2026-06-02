@echo off
REM ===================================================================
REM  ReliabilityRAG — Tek tik launcher
REM  Bu dosyayi cift tikla: Gradio uygulamasi baslar, tarayicida acilir.
REM ===================================================================

title ReliabilityRAG

REM Bu .bat dosyasinin bulundugu klasore gec
cd /d "%~dp0"

REM Conda 'base' ortamini aktive et (Anaconda kuruluysa)
call conda activate base >nul 2>&1

REM HuggingFace indirme hizlandirici (varsa)
set HF_HUB_ENABLE_HF_TRANSFER=1

REM Tarayiciyi otomatik ac — 25 saniye sonra (LLM yuklenme suresi)
start "" /b cmd /c "timeout /t 25 /nobreak >nul && start http://127.0.0.1:7860"

echo.
echo ========================================================
echo  ReliabilityRAG baslatiliyor...
echo  Tarayici ~25 saniye sonra otomatik acilacak.
echo  Manuel acmak icin: http://127.0.0.1:7860
echo.
echo  Kapatmak icin bu pencereyi kapatin (CTRL+C de calisir).
echo ========================================================
echo.

REM Uygulamayi calistir
python app.py

REM Hata varsa pencere kapansin diye bekle
echo.
echo ========================================================
echo  Uygulama durdu.
echo ========================================================
pause
