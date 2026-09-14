@echo off
echo ============================================================
echo   KI-gestuetzte 3D-Stadtmodell-Pipeline
echo   Fallbeispiel: Stadt Darmstadt
echo ============================================================

:: Virtuelle Umgebung aktivieren
echo.
echo Aktiviere virtuelle Umgebung...
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo [FEHLER] Virtuelle Umgebung konnte nicht aktiviert werden.
    echo Stelle sicher dass venv im aktuellen Verzeichnis existiert.
    pause
    exit /b 1
)

echo [OK] Virtuelle Umgebung aktiv.

:: Pipeline ausfuehren
echo.
python run_pipeline.py %*

:: Ergebnis anzeigen
if errorlevel 1 (
    echo.
    echo [FEHLER] Pipeline ist fehlgeschlagen.
) else (
    echo.
    echo [OK] Pipeline erfolgreich abgeschlossen.
)

pause
