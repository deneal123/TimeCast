import sys
from dotenv import load_dotenv, set_key, unset_key
from src import path_to_env
from src.utils.custom_logging import setup_logging
load_dotenv(path_to_env())
log = setup_logging()


def main_menu():
    while True:
        log.info('\n=============================================================\n')
        log.info(' Меню для работы с TimeCast:\n')
        log.info(' 1. Запуск uvicorn')
        log.info(' 2. Запуск pytest')
        log.info(' 3. Выход из меню')
        log.info('\n=============================================================\n')

        choice = input('\nСделайте выбор: ')
        log.info('')

        if choice in ['1', '2']:
            set_key(path_to_env(), 'CHOICE', choice)  # Устанавливаем значение переменной окружения
            break
        elif choice == '3':
            sys.exit()
        else:
            log.info('Выберите между 1-3')


if __name__ == '__main__':
    try:
        main_menu()
    except Exception as ex:
        log.info(ex)
