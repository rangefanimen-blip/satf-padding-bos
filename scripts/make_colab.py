"""Build one self-contained notebook, without publishing or embedding results/data."""
from pathlib import Path
import ast
import hashlib
import json
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
paths = [ROOT / "pyproject.toml", ROOT / "docs/interim/protocol.md"]
for directory in ("src/satf_padding", "tests", "configs"):
    paths.extend(sorted((ROOT / directory).glob("*.py" if directory != "configs" else "*.json")))
paths += [ROOT / "scripts/summarize.py", ROOT / "scripts/verify_results.py"]
embedded = {str(p.relative_to(ROOT)): p.read_text() for p in paths}
checksums = {name: hashlib.sha256(body.encode()).hexdigest() for name, body in embedded.items()}
raw_checksums = {
    ds: json.loads((ROOT / f"results/interim/{ds}_data.json").read_text())["raw_sha256"]
    for ds in ("ml1m", "beauty", "games")
}
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text, collapsed=False):
    source = textwrap.dedent(text).strip() + "\n"
    ast.parse(source)
    cell = nbf.v4.new_code_cell(source)
    if collapsed:
        cell.metadata["cellView"] = "form"
    cells.append(cell)


md("""
    # Zero, PAD и BOS: полный воспроизводимый пилот

    **Один файл → «Выполнить все» → результаты нового обучения, графики и выводы.**

    Код проекта встроен в этот ноутбук. Загружать ZIP, подключать Google Drive,
    клонировать GitHub или вводить токены не нужно. Данные скачиваются с сайтов
    GroupLens и McAuley Lab (около 156 MiB). Нужен доступ в интернет.

    По умолчанию достаточно **CPU**. Выполняются 39 основных и 54 контрольных обучения
    на MovieLens-1M, Amazon Beauty и Amazon Video Games, seeds 17, 29, 43.
    Это бюджет промежуточного отчёта: 3 итерации HOOI и 5 эпох SASRec.
    Время зависит от выданной Colab машины и сети; после запуска покажем фактическое.

    Сохранённых метрик внутри нет: все таблицы и выводы ниже строятся из этого запуска.
    При повторном выполнении ячеек обучения завершённые эксперименты возобновляются;
    повторный запуск первой ячейки создаёт новую папку результатов.
""")
md("""
    ## Что проверяем сейчас и что оставляем на следующий этап

    **Сейчас:** H1 — меняют ли PAD/BOS качество относительно Zero;
    H2 — дешевле ли BOS при сопоставимом качестве с PAD;
    H3 — зависит ли эффект от доступной длины истории;
    H4 — сохраняется ли эффект при изменении веса служебных наблюдений и нормализации.
    H4 — диагностические контроли, а не полное установление причины улучшения.

    **Далее:** сохраняется ли выигрыш после равного подбора гиперпараметров и обучения
    SASRec до сходимости; устойчив ли он к K, локальному окну, seed и временным границам;
    даёт ли сама граница истории эффект при уравненном вкладе историй после Hankel
    и при раздельном включении служебного вклада в обучение и предсказание.
""")
md("""
    ## 1. Установка и встроенный код

    Все исходники и 12 проверок корректности будут сохранены в рабочую папку.
    Уже подходящие версии библиотек сохраняются; точное окружение записывается в результаты.
    `SASREC_DEVICE="cpu"` соответствует режиму отчёта. Значение `auto` использует CUDA,
    если она доступна; измерения такого SASRec нельзя смешивать с CPU-временем отчёта.
""")
code('''
    # @title Настройка запуска и установка зависимостей
    SASREC_DEVICE = "cpu"  # @param ["cpu", "auto"]
    import os, sys, json, hashlib, subprocess, time, platform, zipfile, socket
    from pathlib import Path
    from datetime import datetime, timezone
    NOTEBOOK_STARTED = time.perf_counter()
    socket.setdefaulttimeout(60)
    BASE = Path("/content") if Path("/content").is_dir() else Path.cwd()
    PROJECT = BASE / "satf_padding_single"
    PROJECT.mkdir(parents=True, exist_ok=True)
    os.chdir(PROJECT)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "2"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "numpy>=1.26,<3", "scipy>=1.12,<2", "pandas>=2.1,<3", "torch>=2.2,<3",
                    "psutil>=5.9", "matplotlib>=3.8,<4", "pytest>=8"], check=True)
    print("Рабочая папка:", PROJECT)
''')
code("# @title Встроенные исходники проекта (можно раскрыть)\nPROJECT_FILES = " + repr(embedded)
     + "\nSOURCE_MANIFEST = " + repr(checksums) + "\nRAW_SHA256 = " + repr(raw_checksums), collapsed=True)
code('''
    for relative, body in PROJECT_FILES.items():
        assert hashlib.sha256(body.encode()).hexdigest() == SOURCE_MANIFEST[relative]
        dest = PROJECT / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body, encoding="utf-8")
    sys.path.insert(0, str(PROJECT / "src"))
    os.environ["PYTHONPATH"] = str(PROJECT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    import importlib
    importlib.invalidate_caches()
    # При повторном старте настройки перечитываем именно встроенную версию.
    for name in list(sys.modules):
        if name == "satf_padding" or name.startswith("satf_padding."):
            del sys.modules[name]
    import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
    from IPython.display import display, Markdown, Image, FileLink
    from satf_padding.experiment import source_hash
    DEVICE = "cuda" if SASREC_DEVICE == "auto" and torch.cuda.is_available() else "cpu"
    RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    RUN_ROOT = PROJECT / "runs" / RUN_ID
    MAIN = RUN_ROOT / "main"
    CONTROLS = RUN_ROOT / "controls"
    FIGURES = RUN_ROOT / "figures"
    for folder in (MAIN, CONTROLS, FIGURES):
        folder.mkdir(parents=True, exist_ok=True)
    CONFIG = json.loads(Path("configs/interim.json").read_text())
    DATASETS = CONFIG["datasets"]
    LABELS = {"ml1m": "MovieLens-1M", "beauty": "Amazon Beauty", "games": "Amazon Video Games"}
    MODEL_LABELS = {"popular": "MostPopular", "zero": "Zero", "pad": "PAD", "bos": "BOS", "sasrec": "SASRec (5 epochs)"}
    MODEL_ORDER = list(MODEL_LABELS)
    environment = {"run_id": RUN_ID, "python": sys.version, "platform": platform.platform(),
                   "processor": platform.processor(), "cpu_count": os.cpu_count(), "threads": 2,
                   "sasrec_device": DEVICE, "tensor_device": "cpu", "source_sha256": source_hash(),
                   "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None}
    (RUN_ROOT / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (RUN_ROOT / "requirements-resolved.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    (RUN_ROOT / "source_manifest.json").write_text(json.dumps(SOURCE_MANIFEST, indent=2), encoding="utf-8")

    def run_command(arguments):
        """Stream progress in Colab; preserve the full log; stop on failure."""
        with (RUN_ROOT / "execution.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-u", *map(str, arguments)],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            result = process.wait()
            if result:
                raise RuntimeError(f"Команда завершилась с кодом {result}: {arguments}")

    print("LA-SATF: CPU; SASRec:", DEVICE, "| threads: 2")
    print("Результаты нового запуска:", RUN_ROOT)
    display(pd.DataFrame([environment]))
    print(json.dumps(CONFIG, indent=2))
''')
md("""
    ## 2. Проверка реализации и получение данных

    Проверяем повторяющийся PAD, четыре Hankel-свёртки против плотного эталона,
    folding-in, совпадение полных историй, исключение служебных объектов,
    временное разбиение, ранги и причинную маску SASRec. Ошибка останавливает выполнение.

    Протокол: любая оценка — взаимодействие; повтор пары удаляется с сохранением первого
    события. Общие временные границы — 80-й и 90-й процентили. Каталог и модели обучаются
    только на train. Target — первое событие периода; неизвестный объект не заменяем
    более поздним известным. История берётся на момент предсказания; на test доступны
    прошлые события validation, без переобучения модели. Оценка по полному каталогу
    с исключением всей просмотренной истории и служебных объектов.
""")
code('''
    run_command(["-m", "pytest", "-q"])
    from satf_padding.data import download, SOURCES
    for dataset in DATASETS:
        print(f"Получение {LABELS[dataset]}: {SOURCES[dataset][1]}", flush=True)
        for attempt in range(3):
            try:
                path = download(dataset)
                break
            except Exception:
                if attempt == 2:
                    raise
                print("Загрузка прервалась; повтор через 3 секунды", flush=True)
                time.sleep(3)
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != RAW_SHA256[dataset]:
            raise ValueError(f"Изменился исходный архив {dataset}: проверьте версию данных")
        print(f"  {path.stat().st_size / 2**20:.1f} MiB; SHA-256 совпадает", flush=True)
''')
md("""
    ## 3. Обучение и оценка всех основных моделей

    39 запусков: три кодировки LA-SATF и SASRec по три seed на трёх датасетах,
    плюс детерминированный MostPopular по одному разу. Каждое обучение идёт в отдельном
    процессе. Сохраняются validation/test-ранги, NDCG@10, HR@10, время, RSS и конфигурация.
    После каждой модели появится строка прогресса. Если прервать эту ячейку, её можно
    выполнить повторно: уже завершённые запуски пропускаются.
""")
code('''
    main_started = time.perf_counter()
    run_command(["-m", "satf_padding.experiment", "--config", "configs/interim.json",
                 "--output", MAIN, "--device", DEVICE, "--threads", "2"])
    main_elapsed = time.perf_counter() - main_started
    run_command(["scripts/verify_results.py", "--results", MAIN, "--config", "configs/interim.json"])
    run_command(["scripts/summarize.py", "--results", MAIN])
    summary = pd.read_csv(MAIN / "summary.csv")
    runs = pd.read_csv(MAIN / "runs.csv")
    paired = pd.read_csv(MAIN / "paired.csv")
    stats = {ds: json.loads((MAIN / f"{ds}_data.json").read_text()) for ds in DATASETS}
    data_rows = []
    for ds, s in stats.items():
        data_rows.append({"Датасет": LABELS[ds], "События": s["events"],
                          "Train": s["train_events"], "Каталог train": s["catalog"],
                          "Validation": s["validation"]["evaluated"], "Test": s["test"]["evaluated"],
                          "Cold targets test": s["test"]["cold_target"],
                          "Нет истории test": s["test"]["empty_known_history"],
                          "t1 UTC": pd.to_datetime(s["t1"], unit="s", utc=True),
                          "t2 UTC": pd.to_datetime(s["t2"], unit="s", utc=True)})
    data_table = pd.DataFrame(data_rows)
    display(data_table)
    data_table.to_csv(RUN_ROOT / "dataset_statistics.csv", index=False)
    print(f"Основные запуски с подготовкой данных: {main_elapsed / 60:.1f} мин.")
''')
md("""
    ## 4. H1: качество Zero / PAD / BOS и сравнение с бейзлайнами

    Основная метрика — **NDCG@10**, дополнительно **HR@10**. Для одного target HR@10
    совпадает с Recall@10. В таблице среднее ± SD по трём seed; для MostPopular один запуск.
    Отрезки на графиках — SD, не доверительный интервал. Validation показывается отдельно.
""")
code('''
    def measured_table(frame, metric, deviation=None, order=None):
        order = order or list(dict.fromkeys(frame["model"]))
        table = pd.DataFrame(index=order, columns=[LABELS[ds] for ds in DATASETS])
        for row in frame.to_dict("records"):
            value = f"{row[metric]:.5f}"
            if deviation and pd.notna(row.get(deviation)):
                value += f" ± {row[deviation]:.5f}"
            table.loc[row["model"], LABELS[row["dataset"]]] = value
        return table.rename(index=MODEL_LABELS)

    display(Markdown("**Test NDCG@10, mean ± SD**"))
    display(measured_table(summary, "ndcg", "ndcg_sd", MODEL_ORDER))
    display(Markdown("**Test HR@10, mean**"))
    display(measured_table(summary, "hr", order=MODEL_ORDER))
    display(Markdown("**Validation NDCG@10, mean**"))
    display(measured_table(summary, "val_ndcg", order=MODEL_ORDER))
    display(Image(filename=str(MAIN / "quality.png")))

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.7))
    colors = ["#9ca3af", "#93a4b5", "#147d92", "#e7a13d", "#704e99"]
    for ax, ds in zip(axes, DATASETS):
        group = runs[runs.dataset == ds].groupby("model").hr.agg(["mean", "std"]).loc[MODEL_ORDER]
        ax.bar(range(5), group["mean"], color=colors, yerr=group["std"].fillna(0), capsize=3)
        ax.set_xticks(range(5), [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=35, ha="right")
        ax.set_title(LABELS[ds]); ax.set_ylabel("Test HR@10")
    fig.tight_layout(); fig.savefig(FIGURES / "hr10.png", dpi=160); plt.show(); plt.close(fig)
''')
md("""
    Парное сравнение: для каждого пользователя усредняем метрику по seed, затем
    2000 раз пересэмплируем пользователей. **95% интервалы условны на трёх обучениях**:
    они не охватывают всю неопределённость обучения и не скорректированы за множественные
    сравнения. Положительное среднее само по себе не доказывает устойчивое улучшение.
""")
code('''
    display(paired[paired.contrast.isin(["pad-zero", "bos-zero"])])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3))
    for ax, ds in zip(axes, DATASETS):
        group = paired[(paired.dataset == ds) & paired.contrast.isin(["pad-zero", "bos-zero"])]
        for i, row in enumerate(group.itertuples()):
            ax.plot([row.ci_user_low, row.ci_user_high], [i, i], color="#147d92", linewidth=3)
            ax.scatter([row.delta_ndcg], [i], color="#263649", zorder=3)
        ax.axvline(0, linestyle="--", color="#999999")
        ax.set_yticks(range(len(group)), group.contrast)
        ax.set_xlabel("Delta NDCG@10 and conditional 95% CI")
        ax.set_title(LABELS[ds]); ax.set_ylim(-.5, 1.5)
    fig.tight_layout(); fig.savefig(FIGURES / "paired_intervals.png", dpi=160); plt.show(); plt.close(fig)
''')
md("""
    ## 5. H2: стоимость BOS относительно PAD

    Сопоставляем только три тензорные кодировки на одном CPU. `fit` включает подготовку
    представления и проектора, но не скачивание/чтение данных. Оценка включает вычисление
    scores, маскирование и ранги. RSS — память **всего процесса** при fit+eval,
    измеряемая каждые 20 мс; это не размер модели и не VRAM.
    Порог эквивалентности качества заранее не задан, поэтому «сопоставимое качество»
    нельзя формально подтвердить только по близости средних.
""")
code('''
    cost = summary[summary.model.isin(["zero", "pad", "bos"])][
        ["dataset", "model", "ndcg", "train_seconds", "ms_per_user", "peak_rss_mb"]]
    display(cost)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.7))
    for ax, metric, label in zip(axes, ["train_seconds", "ms_per_user", "peak_rss_mb"],
                                 ["Fit, seconds (mean)", "Evaluation, ms/user (mean)", "Process peak RSS, MiB (max)"]):
        for j, model in enumerate(["zero", "pad", "bos"]):
            group = cost[cost.model == model].set_index("dataset").loc[DATASETS]
            ax.bar(np.arange(3) + (j-1)*.24, group[metric], width=.24,
                   label=MODEL_LABELS[model], color=colors[j+1])
        ax.set_xticks(range(3), ["ML-1M", "Beauty", "Games"])
        ax.set_ylabel(label); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "tensor_cost.png", dpi=160); plt.show(); plt.close(fig)
''')
md("""
    ## 6. H3: эффект по длине доступной истории

    Длина — число известных объектов **на момент предсказания**, обрезанное до K=50.
    Одинаковые пользователи входят в каждый сравниваемый срез. При n≥50 входные
    кодировки совпадают, однако отдельно обученные модели могут выдавать разные scores.
    Над каждым срезом указано число пользователей; редкие полные истории Amazon
    не позволяют делать надёжные выводы об отсутствии эффекта.
""")
code('''
    history_rows = []
    bins = [("1–5", 1, 5), ("6–20", 6, 20), ("21–49", 21, 49), ("50+", 50, 50)]
    for ds in DATASETS:
        for model in ["zero", "pad", "bos"]:
            for seed in CONFIG["seeds"]:
                with np.load(MAIN / f"{ds}_{model}_{seed}_test.npz") as a:
                    for label, lo, hi in bins:
                        mask = (a["lengths"] >= lo) & (a["lengths"] <= hi)
                        history_rows.append({"dataset": ds, "model": model, "seed": seed,
                            "history": label, "n": int(mask.sum()),
                            "ndcg": float(a["ndcg"][mask].mean()) if mask.any() else np.nan})
    history = pd.DataFrame(history_rows).groupby(["dataset", "model", "history"], sort=False).agg(
        n=("n", "first"), ndcg=("ndcg", "mean"), sd=("ndcg", "std")).reset_index()
    history.to_csv(RUN_ROOT / "history_slices.csv", index=False)
    display(history)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.7))
    for ax, ds in zip(axes, DATASETS):
        for model, color in zip(["zero", "pad", "bos"], colors[1:4]):
            group = history[(history.dataset == ds) & (history.model == model)].set_index("history").loc[[b[0] for b in bins]]
            ax.errorbar(range(4), group.ndcg, yerr=group.sd.fillna(0), marker="o", capsize=3,
                        label=MODEL_LABELS[model], color=color)
        ax.set_xticks(range(4), [f"{b[0]}\\nn={int(group.loc[b[0], 'n'])}" for b in bins])
        ax.set_ylabel("Test NDCG@10, mean ± SD"); ax.set_title(LABELS[ds]); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "history_slices.png", dpi=160); plt.show(); plt.close(fig)
''')
md("""
    ## 7. H4: контроль веса служебных записей и нормализации

    Ещё **54 обучения**: шесть вариантов × три датасета × три seed.

    - `pad_balanced`: вес каждого PAD = 1/√q; уравнивает энергию служебных записей
      до Hankel, но не полный вклад истории после преобразования.
    - `start`: один маркер в крайней левой позиции. Он удаляется сдвигом при предсказании,
      поэтому это контроль влияния служебной записи через обучение.
    - `zero_scaled`, `pad_scaled`, `bos_scaled`: нормализация реальных объектов, scaling=0.2.
    - `pad_scaled_frequency`: дополнительно нормируется PAD по его частоте в train.

    Сравнения помогают обнаружить чувствительность к весам и нормировке. Они не изолируют
    все причины: отдельная нормализация вклада историй после Hankel остаётся следующим шагом.
""")
code('''
    controls_started = time.perf_counter()
    run_command(["-m", "satf_padding.experiment", "--config", "configs/ablations.json",
                 "--output", CONTROLS, "--device", "cpu", "--threads", "2"])
    controls_elapsed = time.perf_counter() - controls_started
    run_command(["scripts/verify_results.py", "--results", CONTROLS, "--config", "configs/ablations.json"])
    run_command(["scripts/summarize.py", "--results", CONTROLS])
    ablations = pd.read_csv(CONTROLS / "summary.csv")
    combined = pd.concat([summary[summary.model.isin(["zero", "pad", "bos"])], ablations], ignore_index=True)
    ablation_order = ["zero", "pad", "bos", "pad_balanced", "start", "zero_scaled", "pad_scaled", "bos_scaled", "pad_scaled_frequency"]
    display(measured_table(combined, "ndcg", "ndcg_sd", ablation_order))
    combined.to_csv(RUN_ROOT / "mechanism_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, ds in zip(axes, DATASETS):
        group = combined[combined.dataset == ds].set_index("model").loc[ablation_order]
        ax.barh(range(len(group)), group.ndcg, xerr=group.ndcg_sd.fillna(0), color="#147d92", capsize=3)
        ax.set_yticks(range(len(group)), ablation_order, fontsize=8); ax.invert_yaxis()
        ax.set_xlabel("Test NDCG@10, mean ± SD"); ax.set_title(LABELS[ds])
    fig.tight_layout(); fig.savefig(FIGURES / "mechanism_controls.png", dpi=160); plt.show(); plt.close(fig)
    print(f"Контрольные запуски с подготовкой данных: {controls_elapsed / 60:.1f} мин.")
''')
md("""
    ## 8. Диагностика обучения SASRec

    Показываем train-loss каждой из пяти эпох и качество validation/test.
    Уменьшение train-loss не доказывает сходимость или оптимальность модели.
    Этот короткий запуск — пилотный бейзлайн; он не позволяет заявить превосходство
    над настроенным SASRec. Для дальнейшего сравнения нужны кривые validation
    по эпохам, ранняя остановка и равный бюджет подбора конфигураций.
""")
code('''
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    for ax, ds in zip(axes, DATASETS):
        for seed in CONFIG["seeds"]:
            trial = json.loads((MAIN / f"{ds}_sasrec_{seed}.json").read_text())
            ax.plot(range(1, len(trial["losses"])+1), trial["losses"], marker="o", label=f"seed {seed}")
        ax.set_title(LABELS[ds]); ax.set_xlabel("Epoch"); ax.set_ylabel("Train sampled BCE"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "sasrec_training.png", dpi=160); plt.show(); plt.close(fig)
    display(summary[summary.model == "sasrec"][["dataset", "val_ndcg", "ndcg", "ndcg_sd", "hr", "train_seconds"]])
''')
md("""
    ## 9. Выводы по вашему запуску

    Следующая ячейка подставляет фактически полученные числа. Статус гипотез
    не задан заранее. Положительные пилотные контрасты не считаются окончательным
    подтверждением общего выигрыша метода.
""")
code('''
    lookup = summary.set_index(["dataset", "model"])
    control_lookup = ablations.set_index(["dataset", "model"])
    lines = ["# Выводы текущего запуска", "", "## H1 — качество относительно Zero"]
    for ds in DATASETS:
        for model in ["pad", "bos"]:
            p = paired[(paired.dataset == ds) & (paired.contrast == f"{model}-zero")].iloc[0]
            status = "целиком выше нуля" if p.ci_user_low > 0 else "целиком ниже нуля" if p.ci_user_high < 0 else "включает ноль"
            lines.append(f"- {LABELS[ds]}, {MODEL_LABELS[model]} − Zero: ΔNDCG@10 = {p.delta_ndcg:+.5f}; "
                         f"условный 95% CI [{p.ci_user_low:+.5f}; {p.ci_user_high:+.5f}], {status}.")
    lines += ["", "Интервалы описывают вариацию пользователей при этих трёх seed; поправки за множественные сравнения нет.",
              "", "## H2 — стоимость BOS относительно PAD"]
    for ds in DATASETS:
        p, b = lookup.loc[(ds, "pad")], lookup.loc[(ds, "bos")]
        lines.append(f"- {LABELS[ds]}: fit BOS/PAD = {b.train_seconds/p.train_seconds:.2f}×; "
                     f"peak RSS BOS/PAD = {b.peak_rss_mb/p.peak_rss_mb:.2f}×; "
                     f"ΔNDCG@10 (BOS − PAD) = {b.ndcg-p.ndcg:+.5f}.")
    lines += ["", "Отношение меньше 1 означает меньшую измеренную стоимость BOS. "
              "Эквивалентность качества не проверена: заранее заданного допуска нет.",
              "", "## H3 — длина истории"]
    for ds in DATASETS:
        z, p = lookup.loc[(ds, "zero")], lookup.loc[(ds, "pad")]
        lines.append(f"- {LABELS[ds]}: PAD − Zero на коротких историях = {p.short_ndcg-z.short_ndcg:+.5f} "
                     f"(n={int(z.short_n)}); на полных = {p.full_ndcg-z.full_ndcg:+.5f} (n={int(z.full_n)}).")
        if z.full_n < 100:
            lines.append("  Полных историй меньше 100: оценка этого среза особенно ненадёжна.")
    lines += ["", "Разница средних по срезам — описательный результат; отдельный тест взаимодействия длины и кодировки не выполнен.",
              "", "## H4 — служебные наблюдения и нормировка"]
    for ds in DATASETS:
        p, z = lookup.loc[(ds, "pad")], lookup.loc[(ds, "zero")]
        c = control_lookup.loc[ds]
        lines.append(f"- {LABELS[ds]}: balanced PAD − PAD = {c.loc['pad_balanced', 'ndcg']-p.ndcg:+.5f}; "
                     f"левый маркер − Zero = {c.loc['start', 'ndcg']-z.ndcg:+.5f}; "
                     f"частотная нормировка PAD − единичный вес PAD при scaling=0.2 = "
                     f"{c.loc['pad_scaled_frequency', 'ndcg']-c.loc['pad_scaled', 'ndcg']:+.5f}.")
    lines += ["", "Эти изменения показывают чувствительность к выбранным вмешательствам. "
              "Отделить эффект самой границы от изменения задачи аппроксимации пока нельзя.",
              "", "## Ограничения и следующий этап", "",
              "- Бюджет фиксирован и мал: 3 итерации HOOI, 5 эпох SASRec, без подбора гиперпараметров.",
              "- Дальше: одинаковый бюджет настройки по validation, SASRec до validation-плато, больше seed, K, окон и временных границ.",
              "- Для механизма: уравнять вклад историй после Hankel и отдельно отключать служебный вклад при инференсе одной обученной модели.",
              "- Warm-item протокол исключает cold targets; Amazon 5-core ограничивает популяцию по всему исходному периоду.",
              "- Уже просмотренный test нельзя использовать как независимое подтверждение новых гипотез; нужны заранее выделенные новые временные интервалы."]
    conclusion = "\\n".join(lines)
    (RUN_ROOT / "conclusions.md").write_text(conclusion, encoding="utf-8")
    display(Markdown(conclusion))
''')
md("""
    ## 10. Сохранение результатов

    Архив содержит JSON и NPZ всех 93 запусков, CSV, графики, выводы, конфигурации,
    использованные исходники и версии библиотек. Сырые датасеты и пользовательские ID
    в архив не входят. При завершении Colab-сессии рабочая папка может исчезнуть:
    сохраните архив. При открытии локально появится ссылка на него.
""")
code('''
    from satf_padding.experiment import source_hash as current_source_hash
    assert current_source_hash() == environment["source_sha256"]
    main_count = len(runs)
    control_count = len(pd.read_csv(CONTROLS / "runs.csv"))
    assert main_count == 39 and control_count == 54, (main_count, control_count)
    elapsed = time.perf_counter() - NOTEBOOK_STARTED
    (RUN_ROOT / "timing.json").write_text(json.dumps({"total_wall_seconds": elapsed,
        "main_command_wall_seconds": main_elapsed, "controls_command_wall_seconds": controls_elapsed,
        "main_runs": main_count, "control_runs": control_count}, indent=2), encoding="utf-8")
    archive = PROJECT / f"satf_results_{RUN_ID}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(RUN_ROOT.rglob("*")):
            if path.is_file():
                z.write(path, "results/" + str(path.relative_to(RUN_ROOT)))
        for relative in PROJECT_FILES:
            z.write(PROJECT / relative, "project/" + relative)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(f"Готово: {main_count} основных + {control_count} контрольных обучений.")
    print(f"Общее время, включая скачивание/подготовку/графики: {elapsed / 60:.1f} мин.")
    print(f"Архив: {archive.name} ({archive.stat().st_size / 2**20:.2f} MiB)")
    try:
        from google.colab import files
    except ImportError:
        display(FileLink(str(archive.relative_to(PROJECT))))
    else:
        files.download(str(archive))
''')
md("""
    ## Источники и воспроизводимость

    - [LA-SATF: Frolov & Oseledets, IEEE Access, 2023](https://doi.org/10.1109/ACCESS.2023.3234863).
    - [SASRec: Kang & McAuley, ICDM 2018](https://arxiv.org/abs/1808.09781).
    - [MovieLens-1M, GroupLens](https://grouplens.org/datasets/movielens/1m/).
    - [Amazon 2014, McAuley Lab](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html).

    Собственные реализации по математическим постановкам; точное воспроизведение
    чисел оригинальных статей не заявляется. SHA-256 исходных данных и кода проверяются.
    Отличия библиотек, аппаратуры и численных алгоритмов могут менять результаты;
    в каждом запуске сохраняется его окружение. Публикация проекта этим ноутбуком не выполняется.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"name": "satf_padding_bos_all_in_one.ipynb", "provenance": [], "toc_visible": True},
})
nbf.validate(nb)
dest = ROOT / "notebooks/satf_padding_bos_all_in_one.ipynb"
nbf.write(nb, dest)
print(f"Created {dest}: {len(cells)} cells, {dest.stat().st_size:,} bytes")
