import time
import subprocess
import sys
import pkg_resources

def install_requirements():
    # Чтение requirements.txt
    with open('requirements.txt', 'r') as file:
        requirements = file.readlines()
    
    # Проверка на установленные пакеты
    missing_packages = []
    for requirement in requirements:
        package = requirement.strip()
        try:
            pkg_resources.require(package)
        except pkg_resources.DistributionNotFound:
            missing_packages.append(package)
    
    # Установка только недостающих пакетов
    if missing_packages:
        print(f"Устанавливаются недостающие пакеты: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        except subprocess.CalledProcessError as e:
            print(f"Ошибка при установке зависимостей: {e}")
            sys.exit(1)
    else:
        print("Все зависимости уже установлены.")

install_requirements()

# Запуск скриптов
subprocess.run(['python', 'C:/AI/scripts/extract_text.py'])
time.sleep(1)
subprocess.run(['python', 'C:/AI/scripts/augment_text.py'])
time.sleep(1)
subprocess.run(['python', 'C:/AI/scripts/create_dataset.py'])
time.sleep(1)

print("Все скрипты выполнены.")
