



pdfinfo -listenc
IF %ERRORLEVEL% NEQ 0 exit /B 1
pdfunite --help
IF %ERRORLEVEL% NEQ 0 exit /B 1
pdftocairo --help
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0
