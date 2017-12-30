@echo off


::git log -1 --format="%%h" > .run
echo %DATE%_%TIME%>.run
set /P RUN=<.run
del .run

set RUN=%RUN::=%
set RUN=%RUN: =0%
set RUN_PATH=T:\tf\hw3\run_%RUN:~0,-3%

echo logs: %RUN_PATH%
md %RUN_PATH%

python hw3.py
::python 1.py