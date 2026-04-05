#!/bin/bash

# Переходим в директорию, где лежит этот скрипт (т.е. в корень проекта TIMECODER)
cd "$(dirname "$0")"

# Подтягиваем nvm/node для yt-dlp JS runtime при запуске из Finder/Terminal
export NVM_DIR="$HOME/.nvm"
if [ -s "$NVM_DIR/nvm.sh" ]; then
    . "$NVM_DIR/nvm.sh"
fi

# Не даем PyTorch падать на неподдержанных MPS-операциях в embedding-стеке
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "========================================="
echo "   Запускаю Timecoder AI Assistant...    "
echo "========================================="

# Ищем и останавливаем любой уже запущенный сервер на порту 8000
PIDS=$(lsof -ti :8000)
if [ -n "$PIDS" ]; then
    echo "Найден старый процесс на порту 8000 (PID: $PIDS). Останавливаю..."
    kill -9 $PIDS
    sleep 1
fi

# Фоновый процесс: ждем 2 секунды, пока запустится сервер, и открываем браузер (только для Mac)
(sleep 2 && open "http://localhost:8000/") &

# Активируем виртуальное окружение и запускаем FastAPI сервер
source venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
uvicorn src.api.app:app

# Если сервер остановится (например, при нажатии Ctrl+C), скрипт завершит работу
