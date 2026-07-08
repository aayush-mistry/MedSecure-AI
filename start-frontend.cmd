@echo off
cd /d "%~dp0frontend"
call npm.cmd run dev -- --host 0.0.0.0
