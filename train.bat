@echo off

rem Запуск скрипта для извлечения текста из PDF
python C:/AI/scripts/train_llama.py
timeout /t 1 > nul

echo
pause