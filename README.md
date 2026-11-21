# Zero, PAD и BOS в LA-SATF: промежуточный пилот

Проверяем влияние заполнения коротких историй в тензорной последовательной рекомендательной модели. Реализованы Zero, повторяющийся служебный объект PAD и единичная метка BOS; внешние бейзлайны — SASRec и MostPopular.

В репозитории находятся **пилотные эксперименты с малым бюджетом**: 39 основных и 54 контрольных запуска на MovieLens-1M, Amazon Beauty 2014 и Amazon Video Games 2014. Это проверка работоспособности и предварительных эффектов; результаты не доказывают превосходство над настроенным SASRec. Увеличение моделей, бюджета обучения и расширение данных — дальнейшая работа.

[Промежуточный отчёт (PDF)](docs/interim/interim_report.pdf) — актуальная версия с результатами 93 запусков в Colab.

## Один ноутбук Colab

Скачайте [satf_padding_bos_all_in_one.ipynb](notebooks/satf_padding_bos_all_in_one.ipynb), загрузите его в Colab и выполните все ячейки. По умолчанию достаточно CPU. Код встроен: отдельный ZIP, GitHub-токен и Google Drive не нужны.

Ноутбук сам устанавливает зависимости, получает три датасета, проверяет SHA-256, выполняет проверки реализации, запускает основной пилот и контроли, строит таблицы/графики, формирует выводы из новых измерений и предлагает сохранить результаты архивом. Готовые метрики в ноутбук не подставляются.

Бюджет основного сравнения: K=50, ранги LA-SATF (12,12,2,2), окно L=5, **3 итерации HOOI**; SASRec dim=32, 1 блок, **5 эпох**. Seeds: 17, 29, 43; MostPopular запускается один раз. Подбор гиперпараметров в пилоте не выполняется.

Исходные архивы прочитаны целиком, но обучение использует только train-часть: у LA-SATF последние 50 взаимодействий пользователя, у SASRec последние 50 пар next-item. Это не обучение на неограниченных полных историях.

## Гипотезы

- **H1:** PAD/BOS улучшают качество относительно Zero — NDCG@10, HR@10 и парные разности.
- **H2:** BOS дешевле PAD при сопоставимом качестве — время обучения/оценки, RSS и качество. Формальный допуск эквивалентности качества не задан.
- **H3:** эффект зависит от доступной длины истории — одинаковые пользователи в срезах коротких и полных историй.
- **H4:** эффект связан также с весами служебных наблюдений и нормализацией — диагностические контроли, без полного причинного разделения механизмов.

Следующий этап: увеличение бюджета и проверка сходимости, одинаковый бюджет настройки по validation, новые временные интервалы/более крупные наборы и отдельная проверка вклада границы истории. Уже просмотренный test не считается независимым подтверждением новых гипотез.

## Код

- [tensor.py](src/satf_padding/tensor.py): `encode()` для Zero/PAD/BOS, разреженные Hankel-свёртки, HOOI и folding-in. Повторяющиеся PAD учитываются во всех позициях. Полные входные истории совпадают для трёх кодировок.
- [sasrec.py](src/satf_padding/sasrec.py): собственный PyTorch-бейзлайн SASRec. Варианты PAD/BOS внутри него не исследуются.
- [data.py](src/satf_padding/data.py): получение данных, дедупликация, временное разбиение и train-каталог.
- [evaluation.py](src/satf_padding/evaluation.py): full-catalog ранги с исключением просмотренных и служебных объектов.
- [experiment.py](src/satf_padding/experiment.py): отдельный процесс на запуск, метрики, время/RSS, контрольные суммы и возобновление.
- [configs](configs): точные пилотные бюджеты; [tests](tests): проверки плотным эталоном, протокола и масок.

Это независимые реализации по [Frolov и Oseledets, 2023](https://doi.org/10.1109/ACCESS.2023.3234863) и [Kang и McAuley, 2018](https://arxiv.org/abs/1808.09781). Точное воспроизведение чисел оригинальных статей не заявляется.

## Получение данных и протокол

Исходные архивы автоматически скачиваются в `data/raw` с сайтов владельцев:

| Датасет | Источник |
|---|---|
| MovieLens-1M | [GroupLens](https://files.grouplens.org/datasets/movielens/ml-1m.zip) |
| Amazon Beauty 2014, 5-core | [McAuley Lab](https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/reviews_Beauty_5.json.gz) |
| Amazon Video Games 2014, 5-core | [McAuley Lab](https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/reviews_Video_Games_5.json.gz) |

Любая оценка считается взаимодействием; для пары user–item сохраняется самое раннее событие. Глобальные временные границы — квантили 0.8 и 0.9 после дедупликации. Каталог и модели обучаются только на train. Цель — первое новое событие каждого периода. Cold-target не заменяется последующим известным объектом. История доступна на момент предсказания, на test включает прошлые события validation без дообучения модели. Подробности и ограничения — в [протоколе](docs/interim/protocol.md).

Сырые данные и пользовательские ID в репозиторий не включены. Сохранены агрегированные статистики, контрольные суммы и массивы рангов/длин истории. Готовая Amazon 5-core и warm-item исключения ограничивают популяцию эксперимента.

## Воспроизведение

Проверено на Python 3.12, macOS arm64, CPU. Основные зависимости перечислены в `pyproject.toml`; версии среды измерений — в `requirements.txt`. Пример установки и запуска:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -e .
python -m pytest -q
python -m satf_padding.experiment --output results/reproduced
python scripts/summarize.py --results results/reproduced
python scripts/verify_results.py --results results/reproduced
```

Контрольные эксперименты:

```bash
python -m satf_padding.experiment --config configs/ablations.json --output results/reproduced_ablations
python scripts/summarize.py --results results/reproduced_ablations
python scripts/verify_results.py --config configs/ablations.json --results results/reproduced_ablations
```

При повторе пропускаются завершённые результаты с теми же конфигурацией, SHA-256 исходников и разбиения. Для другого окружения, устройства или изменения настроек используйте новую выходную папку. `--force` явно разрешает перезапись. Отдельная GPU-разработка в этот пилотный набор не включена.

## Сохранённые результаты и материалы

- [Основное сравнение](results/interim/summary.csv), [каждый seed](results/interim/runs.csv), [парные разности](results/interim/paired.csv).
- [Контрольные эксперименты](results/ablations/summary.csv).
- [Исходный proposal](docs/proposal.pdf) и [материалы proposal](docs/README.md).

Время обучения исключает скачивание/чтение данных. Peak RSS относится ко всему CPU-процессу, а не к размеру модели или GPU VRAM. Погрешности графиков — SD по seed. Bootstrap-интервалы условны на трёх обучениях, без поправки за множественные сравнения.

Проверить сохранённые результаты **без обучения и скачивания данных**:

```bash
python scripts/verify_results.py
python scripts/verify_results.py --config configs/ablations.json --results results/ablations
```

Пересобрать таблицы, графики, анонимный промежуточный PDF и самостоятельный ноутбук:

```bash
python scripts/summarize.py
python scripts/summarize.py --results results/ablations
python scripts/build_report.py
python scripts/make_colab.py
```

Команды выше пересобирают отчёт по сохранённым локальным результатам в `docs/interim/report.pdf`. Готовый промежуточный отчёт с результатами Colab опубликован отдельно: [`interim_report.pdf`](docs/interim/interim_report.pdf). В Colab прошли 12 тестов; метрики всех 93 запусков сверены с сохранёнными test-рангами. Полноценные эксперименты на увеличенных моделях ещё не выполнены.
