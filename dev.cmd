@echo off
REM Fast development CSS build script for Windows
REM Optimized for rapid Tailwind CSS compilation

echo Starting fast Tailwind CSS build...
echo.

REM Set environment variables for faster builds
set NODE_ENV=development
set TAILWIND_DISABLE_TOUCH=true
set TAILWIND_MODE=watch

REM Clear node module resolution cache
set NODE_OPTIONS=--max-old-space-size=4096

REM Run the fast dev script
npm run dev:css:fast
