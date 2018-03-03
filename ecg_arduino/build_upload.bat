@echo off

REM https://arduino.stackexchange.com/questions/8987/build-arduino-with-windows-command-line
REM https://github.com/keyboardio/Kaleidoscope/wiki/Command-Line-Upload-Instructions-for-Windows
REM https://www.nongnu.org/avrdude/

REM Haven't investigated the full build process as yet.
REM Probably simpler just to create a .INO file? But:
REM http://www.visualmicro.com/page/User-Guide.aspx?doc=INOs-and-CPPs.html

REM Here we go:
REM https://playground.arduino.cc/Code/WindowsCommandLine

REM ===========================================================================
REM Path to script
REM ===========================================================================
REM https://stackoverflow.com/questions/3827567/how-to-get-the-path-of-the-batch-script-in-windows
set SCRIPTDIR=%~dp0
set SCRIPTDIR=%SCRIPTDIR:~0,-1%

REM ===========================================================================
REM Other paths
REM ===========================================================================
set ABUILDDIR=%SCRIPTDIR%\abuild
set OLDPATH=%PATH%
set PATH=%OLDPATH%;%ABUILDDIR%
set ABUILD=abuild.bat
set SRCDIR=%SCRIPTDIR%\src
set SRCJOINEDDIR=%SCRIPTDIR%\src_joined

REM Get #define variables
REM "%ARDUINO_PATH%\hardware\tools\avr\bin\avr-g++.exe" -dM -E - < NUL > tmp_cpp_defines.txt

REM ===========================================================================
REM The main bits
REM ===========================================================================

REM %ABUILD% -v %SRCDIR%\ecg.cpp || goto error
%ABUILD% -v %SRCJOINEDDIR%\ecg_all.cpp || goto error

REM ===========================================================================
REM Finishing up
REM ===========================================================================

echo "Finished."
set ERRORLEVEL=0
goto success

:error
echo Failed with errorlevel %ERRORLEVEL%.
REM and fall through...

:success
set PATH=%OLDPATH%
exit /b
REM exit /b preserves errorlevel
