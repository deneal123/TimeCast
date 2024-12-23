import os


def set_pythonpath():
    # Получаем текущую директорию
    current_dir = os.getcwd()
    print(f"Текущая директория проекта: {current_dir}")
    # Получаем текущие значения PYTHONPATH
    pythonpath = os.environ.get('PYTHONPATH', '')
    # Проверяем, есть ли текущая директория в PYTHONPATH
    if current_dir not in pythonpath:
        # Добавляем текущую директорию в PYTHONPATH
        new_pythonpath = f"{current_dir}:{pythonpath}" if pythonpath else current_dir
        os.environ['PYTHONPATH'] = new_pythonpath
        print(f"PYTHONPATH: {new_pythonpath}")
    else:
        print(f"PYTHONPATH: {current_dir}")