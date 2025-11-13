@echo off
setlocal enabledelayedexpansion

echo PPMD Order Test Results > log.txt
echo ======================= >> log.txt
echo. >> log.txt

set INPUT=..\book1
set MAX_JOBS=7

REM Clean up any old lock and compressed files
del temp_*.lock 2>nul
del temp_*.compressed 2>nul

REM Loop through orders 2-25
set ORDER=2
:startloop
if %ORDER% GTR 25 goto alldone

REM Wait while we have MAX_JOBS running
:waitloop
set /a JOBS=0
for %%f in (temp_*.lock) do set /a JOBS+=1
if !JOBS! GEQ %MAX_JOBS% (
    timeout /t 1 /nobreak >nul
    goto waitloop
)

REM Create lock file FIRST, then start compression job
echo %ORDER% > temp_%ORDER%.lock
echo Starting order %ORDER%...
start /min cmd /c "coder e %INPUT% temp_%ORDER%.compressed %ORDER% & del temp_%ORDER%.lock"

REM Small delay to ensure lock file is written
timeout /t 0 /nobreak >nul

set /a ORDER+=1
goto startloop

:alldone
REM Wait for all jobs to complete
echo Waiting for all jobs to complete...
:finalwait
set /a REMAINING=0
for %%f in (temp_*.lock) do set /a REMAINING+=1
if !REMAINING! GTR 0 (
    echo !REMAINING! jobs remaining...
    timeout /t 2 /nobreak >nul
    goto finalwait
)

REM Collect results
echo.
echo Collecting results...
for /L %%o in (2,1,25) do (
    if exist temp_%%o.compressed (
        for %%A in (temp_%%o.compressed) do (
            echo Order %%o: %%~zA bytes >> log.txt
        )
        del temp_%%o.compressed
    ) else (
        echo Order %%o: FAILED >> log.txt
    )
)

echo. >> log.txt
echo Testing complete! >> log.txt
echo.
echo Results written to log.txt
type log.txt
