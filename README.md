# свэн-тим Hack&Change 2025

Наш первый хакатон:)

29.11.2025 Трек ML / Web: классификация тональности текстов от Правительство Москвы

## Участники команды

- Акаев Адам Баудинович - фронтенд
- Фурсов Владимир Евгеньевич - обучение модели
- Швецов Георгий Владимирович - бэкенд

## Информация

Все кнопки работают, всё работает! Тестировали на файле test.csv

Размещенное приложение: http://46.173.26.112:5000/

Ссылка на GoogleDisk с видео: https://drive.google.com/drive/folders/16Sqkzvhlr3CQpLUP3ZrxGa52WIkN3k7_?usp=sharing

Код для обучения модели лежит в папке "Тренировка модели". Использовали Kaggle notebook!!

Был использован Docker. Всё для сбора образа уже готово. В главе "Запуск проекта" всё описано.

Результаты в папке "Результаты"

Версия python 3.12

Список библиотек:
- --index-url https://download.pytorch.org/whl/cpu
- --extra-index-url https://pypi.org/simple
- blinker==1.9.0
- certifi==2025.11.12
- charset-normalizer==3.4.4
- click==8.3.1
- colorama==0.4.6
- filelock==3.20.0
- Flask==3.1.2
- fsspec==2025.10.0
- huggingface-hub==0.36.0
- idna==3.11
- itsdangerous==2.2.0
- Jinja2==3.1.6
- joblib==1.5.2
- MarkupSafe==3.0.3
- mpmath==1.3.0
- networkx==3.6
- numpy==2.3.5
- packaging==25.0
- pandas==2.3.3
- python-dateutil==2.9.0.post0
- pytz==2025.2
- PyYAML==6.0.3
- regex==2025.11.3
- requests==2.32.5
- safetensors==0.7.0
- scikit-learn==1.7.2
- scipy==1.16.3
- six==1.17.0
- sympy==1.13.3
- threadpoolctl==3.6.0
- tokenizers==0.22.1
- torch==2.9.1
- tqdm==4.67.1
- transformers==4.57.3
- typing_extensions==4.15.0
- tzdata==2025.2
- urllib3==2.5.0
- Werkzeug==3.1.3



## Запуск проекта
Сначала нужно склонировать наш репозиторий. В нём должно быть всего достаточно. В папке "model" уже лежит модель.

Список зависимостей/библиотек в файле requirements.txt

```
pip install -r requirements.txt
```

Чтобы запустить этот проект локально, нужно докачать библиотеки и написать в терминал:
```python
  python main.py
```
ЛИБО использовать Docker и написать в папке проекта:
```PowerShell
  docker compose up --build
```

Если потребуется ускорить, то можно заменить библиотеку torch на её gpu версию:

```
pip uninstall torch -y
```

```
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```