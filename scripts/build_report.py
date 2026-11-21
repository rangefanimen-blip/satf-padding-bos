"""Rebuild the anonymous intermediate report from verified result artifacts."""
from pathlib import Path
import json
import os
from datetime import datetime, timezone
from html import escape
import pandas as pd
import matplotlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'docs/interim'; OUT.mkdir(exist_ok=True)
fontdir=Path(matplotlib.get_data_path())/'fonts/ttf'
for name, file in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf'),('Italic','DejaVuSans-Oblique.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(fontdir/file)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Italic')
INK=colors.HexColor('#182d3e'); TEAL=colors.HexColor('#126a7c')
styles={
 'title':ParagraphStyle('title',fontName='Bold',fontSize=19,leading=23,textColor=INK,spaceAfter=12),
 'h1':ParagraphStyle('h1',fontName='Bold',fontSize=12,leading=15,textColor=TEAL,spaceBefore=10,spaceAfter=6,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='Bold',fontSize=10,leading=13,textColor=INK,spaceBefore=7,spaceAfter=4,keepWithNext=True),
 'p':ParagraphStyle('p',fontName='Body',fontSize=9.1,leading=13,textColor=INK,spaceAfter=6),
 'small':ParagraphStyle('small',fontName='Body',fontSize=7.8,leading=10.4,textColor=INK,spaceAfter=5),
 'cell':ParagraphStyle('cell',fontName='Body',fontSize=7.6,leading=10,textColor=INK),
 'formula':ParagraphStyle('formula',fontName='Body',fontSize=10,leading=16,textColor=INK,spaceBefore=7,spaceAfter=9,leftIndent=10),
}
story=[]; source=[]
def p(text,style='p'):
    story.append(Paragraph(text,styles[style])); source.append(text)
def h(text): p(text,'h1')
def sub(text):p(text,'h2')
def page(): story.append(PageBreak());source.append('\n---\n')
def table(rows,widths=None):
    values=[[Paragraph(escape(str(x)),styles['cell']) for x in row] for row in rows]
    t=Table(values,colWidths=widths or [499/len(rows[0])]*len(rows[0]),repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf2f4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.7,TEAL),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d9e1e5'))]))
    story.append(t); story.append(Spacer(1,7))
    source.append('<pre>'+escape('\n'.join(' | '.join(map(str,row)) for row in rows))+'</pre>')
def fig(filename,caption,width=499,height=145):
    story.append(Image(str(ROOT/'results/interim'/filename),width=width,height=height));p(caption,'small')
def num(x):return f'{x:,.0f}'.replace(',',' ')

names={'ml1m':'MovieLens-1M','beauty':'Amazon Beauty','games':'Amazon Video Games'}
summary=pd.read_csv(ROOT/'results/interim/summary.csv')
abl=pd.read_csv(ROOT/'results/ablations/summary.csv')
paired=pd.read_csv(ROOT/'results/interim/paired.csv')
stats={ds:json.loads((ROOT/f'results/interim/{ds}_data.json').read_text()) for ds in names}
lookup=summary.set_index(['dataset','model'])
def val(ds,m):
    r=lookup.loc[(ds,m)]
    return f'{r.ndcg:.5f}' if pd.isna(r.ndcg_sd) else f'{r.ndcg:.5f} ± {r.ndcg_sd:.5f}'

p('Паддинг и метка начала истории<br/>в тензорных последовательных рекомендациях','title')
p('Промежуточный отчёт · 21 ноября 2026','small')
sub('Аннотация')
p('Исследуем кодирование неполной истории в LA-SATF: нулевой префикс (Zero), повторяемый служебный объект (PAD) и единственную метку начала (BOS). Подготовлены собственные реализации, единый временной протокол и начальные эксперименты на MovieLens-1M, Amazon Beauty и Amazon Video Games. Сравнение выполняется по полному каталогу с тремя инициализациями. Дополнительные контрасты проверяют вклад служебных наблюдений и нормировки. Полученные результаты относятся к малому фиксированному вычислительному бюджету и не устанавливают превосходство метода при оптимальной настройке.')
h('1. Введение')
p('Последовательные рекомендательные системы предсказывают следующий объект по прошлым действиям пользователя. Истории имеют разную длину, тогда как модель обычно получает окно фиксированного размера. В тензорной модели нулевая позиция обозначает отсутствие наблюдения. Такое представление не содержит отдельного объекта, который явно обозначает незаполненный префикс.')
p('Цель проекта — проверить, меняют ли PAD и BOS качество ранжирования следующего нового объекта, как эффект зависит от доступной длины истории и какой вычислительной ценой он достигается. Важно разделить информацию о границе истории и изменение самой задачи аппроксимации: добавление единиц служебного объекта меняет вклад пользователя в тензорную функцию потерь.')
h('2. Обзор литературы')
table([
 ['Подход','Возможности','Ограничение для нашей задачи'],
 ['FPMC [1]','Сочетает персональные предпочтения и переходы между объектами.','Короткий марковский контекст; не исследует кодировки префикса LA-SATF.'],
 ['SASRec [2]','Причинное self-attention выбирает релевантные события истории.','Нейросетевое обучение и настройка; padding в attention не равнозначен наблюдению в тензоре.'],
 ['BERT4Rec [3]','Двунаправленный контекст и восстановление маскированных объектов.','Другая обучающая задача; прямой перенос MASK/BOS в Tucker-модель не обоснован.'],
 ['LA-SATF [4]','Hankel-представление, локальные позиционные веса и Tucker-разложение.','В исходной постановке нулевой префикс; альтернативный padding оставлен для дальнейшей работы.'],
 ],[80,202,217])
p('Пробел проекта — контролируемое сравнение Zero/PAD/BOS именно в LA-SATF с корректным учётом повторяющегося PAD. В отличие от сравнения разных архитектур, основные три варианта сохраняют один алгоритм обучения. Работа Ji et al. [5] также показывает риск утечки при игнорировании общей временной шкалы; поэтому используем глобальное временное разбиение.')

page();h('3. Постановка задачи')
p('Даны события (u, i, t). После удаления повторных пар остаётся последовательность первых взаимодействий пользователя. В момент τ доступна история H<sub>u</sub>(τ) объектов из обучающего каталога I<sub>train</sub>. Требуется упорядочить I<sub>train</sub> ∖ H<sub>u</sub>(τ) и предсказать первое новое взаимодействие в соответствующем периоде. Тексты, изображения, рейтинги как численные признаки и демография не используются.')
p('n<sub>u</sub>(τ) = min(K, |H<sub>u</sub>(τ)|), &nbsp; q<sub>u</sub> = K − n<sub>u</sub>(τ).','formula')
p('При n&lt;K кодировки одной истории имеют вид:','p')
table([['Вариант','Префикс + реальные события'],['Zero','[0, …, 0, i₁, …, iₙ]'],['PAD','[PAD, …, PAD, i₁, …, iₙ]'],['BOS','[0, …, 0, BOS, i₁, …, iₙ]']],[90,409])
p('При n=K служебные записи отсутствуют и все входы совпадают. Длинные истории обрезаются до последних K объектов. Ноль является отсутствием наблюдения, PAD/BOS — наблюдениями дополнительного служебного объекта. Служебный объект всегда исключён из рекомендаций.')
h('4. Предлагаемое решение')
sub('4.1. Обучение и разреженные операции')
p('Пусть X<sup>(c)</sup><sub>uip</sub> — позиционный тензор кодировки c. Для каждого позиционного вектора строится Hankel-матрица размера L×S, где S=K−L+1, H(x)<sub>ab</sub>=x<sub>a+b</sub>. Реализовано HOOI-обучение следующей задачи [4]:')
p('Y<sup>(c)</sup> = H(DX<sup>(c)</sup>) ×₃ Aᵀ,<br/>min ‖Y<sup>(c)</sup> − G ×₁ U ×₂ V ×₃ W<sub>L</sub> ×₄ W<sub>S</sub>‖²<sub>F</sub>.','formula')
p('D задаёт диагональные веса объектов. A — нижнетреугольная матрица локальных весов. Факторы имеют ортонормированные столбцы. Каждая итерация последовательно обновляет четыре фактора по ведущим левым сингулярным подпространствам сжатых развёрток. Реализация хранит K разреженных матриц «пользователь–объект» и суммирует вклад каждой позиции PAD, не создавая полный четырёхмерный тензор. Малые плотные развёртки и матрицы Грама используются для обновления подпространств.')
sub('4.2. Предсказание и самостоятельный вклад')
p('История сначала кодируется в K позициях, затем сдвигается на одну позицию влево, освобождая последнюю под прогноз. Новый BOS после сдвига не вставляется. Используем folding-in через обученные факторы объектов и позиционные проекторы [4]. При фиксированной модели одинаковые входы дают одинаковые scores. Отдельно обученные Zero/PAD/BOS могут различаться даже на полных историях из-за изменения общей обучающей выборки.')
p('Самостоятельный вклад: реализация трёх кодировок и инференса, разреженных свёрток с повторяющимися служебными наблюдениями, проверки плотным эталоном, единый протокол и диагностические абляции. Код написан самостоятельно по математической постановке; это не заявление о точном воспроизведении чисел авторской статьи.')

page();h('5. Данные, гипотезы и экспериментальный сетап')
sub('5.1. Данные и предобработка')
rows=[['Датасет','События','Пользователи','Объекты','Train-события']]
for ds,s in stats.items():rows.append([names[ds],num(s['events']),num(s['users']),num(s['items']),num(s['train_events'])])
table(rows,[125,90,93,81,110])
p('MovieLens-1M [6] содержит оценки фильмов, Amazon Beauty и Video Games [7] — отзывы о товарах. Используем версии Amazon 2014 5-core. Каждая оценка, включая низкую, преобразуется в бинарный факт взаимодействия; цель — следующее действие, а не положительное отношение. Для каждой пары пользователь–объект сохраняем первое событие. Повторная k-core-фильтрация не выполняется. Готовый 5-core Amazon ограничивает популяцию по всему исходному периоду, что остаётся ограничением данных.')
p('Сортировка: timestamp, затем строковые ID пользователя и объекта. При одинаковом timestamp порядок по ID — детерминированное соглашение, а не известный реальный порядок действий. Все три исходных файла прочитаны полностью; их SHA-256 и статистики сохранены рядом с результатами.')
sub('5.2. Точное временное разбиение')
p('После дедупликации берём нижние эмпирические квантили времени 0.8 и 0.9: t₁ и t₂. Train: t&lt;t₁; validation: t₁≤t&lt;t₂; test: t≥t₂. Все события с одинаковым временем остаются вместе. Модель и словарь объектов обучаются только на train. На test модель не переобучается на validation, но история пользователя включает уже доступные события до t₂.')
rows=[['Датасет','t₁ (UTC)','t₂ (UTC)','Val / test, пользователей']]
for ds,s in stats.items():rows.append([names[ds],datetime.fromtimestamp(s['t1'],timezone.utc).strftime('%Y-%m-%d\n%H:%M:%S'),datetime.fromtimestamp(s['t2'],timezone.utc).strftime('%Y-%m-%d\n%H:%M:%S'),f"{s['validation']['evaluated']} / {s['test']['evaluated']}"])
table(rows,[125,116,116,142])
p('В каждом периоде оцениваем первое новое событие пользователя. Если target отсутствует в train-каталоге, исключаем его и не заменяем более поздним известным объектом. Исключаем также случаи без известной истории. История состоит только из доступных к прогнозу известных объектов; именно по ней считаем длину. Кандидаты — весь train-каталог за вычетом всей доступной истории, включая часть за пределами хвоста K.')
rows=[['Датасет','Все test-targets','Cold-target','Нет известной истории','Оценено']]
for ds,s in stats.items():
    t=s['test'];rows.append([names[ds],t['targets'],t['cold_target'],t['empty_known_history'],t['evaluated']])
table(rows,[125,94,88,106,86])
p('Счётчики исключений применяются последовательно и не пересекаются. Результаты относятся к warm-item задаче и не оценивают полный cold start. Разные наборы пользователей в validation и test отражают временной сдвиг.','small')

page();sub('5.3. Гипотезы, бюджет и метрики')
table([['Гипотеза','Проверка'],['H1: PAD/BOS улучшают качество относительно Zero.','Парные разности NDCG@10 на трёх датасетах и трёх seed.'],['H2: BOS дешевле PAD при сопоставимом качестве.','Время fit и оценки, peak RSS процесса, метрики качества.'],['H3: эффект зависит от доступной длины истории.','Одинаковые тестовые пользователи в срезах n<K и n=K.'],['H4: эффект нельзя объяснять только границей истории.','Нормированный PAD, левый маркер и включение нормировки объектов.']],[230,269])
p('Основная метрика — NDCG@10. При одном target она равна 1/log₂(r+1), если r≤10, иначе 0. Дополнительно HR@10=1[r≤10]. Сначала усредняем по пользователям, затем по seed. Равные scores разрешаются по индексу объекта. Negative sampling применяется только при обучении SASRec; оценка всех моделей выполняется по полному допустимому каталогу.')
table([['Параметр','LA-SATF Zero / PAD / BOS','SASRec'],['Окно истории','K=50','K=50'],['Размер модели','Ранги (12,12,2,2), L=5','dim=32, 1 блок, 1 голова'],['Обучение','3 итерации HOOI','5 эпох Adam, lr=0.001'],['Остальное','A[a,b]=(a−b+1)⁻¹; scaling=1','dropout=0.2, batch=128; BCE'],['Инициализации','17, 29, 43','17, 29, 43']],[105,197,197])
p('На модель задана одна фиксированная конфигурация, поиск гиперпараметров не проводился. Бюджет вариантов LA-SATF одинаков. Три итерации HOOI и пять эпох SASRec не означают одинаковую сходимость или wall-clock бюджет. MostPopular использует train-частоты и служит диагностическим ориентиром. Результаты SASRec отражают ранний этап обучения и не характеризуют его потенциальное качество при настройке.')
sub('5.4. Корректность и контрольные эксперименты')
p('Автоматические тесты проверяют все четыре разреженные свёртки относительно плотного четырёхмерного эталона для Zero/PAD/BOS и левого маркера; folding-in относительно плотного проектора; сохранение повторов PAD; совпадение входов при полной истории; отсутствие нового BOS после сдвига; маскирование просмотренного; причинность attention; конечность scores и определение NDCG.')
p('Абляции имеют те же seed и бюджет HOOI. PAD/sqrt(q) уравнивает квадрат нормы служебных записей до Hankel-преобразования. Левый маркер ставится в начало пустого префикса и удаляется при сдвиге инференса, поэтому служит контролем изменения обучения. При scaling=0.2 нормируются реальные объекты; дополнительный вариант нормирует и PAD по его частоте. Эти вмешательства не дают полной причинной идентификации и рассматриваются совместно.')

page();h('6. Текущие результаты и выводы')
sub('6.1. Качество при фиксированном начальном бюджете')
rows=[['Модель','MovieLens-1M','Amazon Beauty','Amazon Video Games']]
for m,label in [('popular','MostPopular'),('zero','LA-SATF Zero'),('pad','LA-SATF PAD'),('bos','LA-SATF BOS'),('sasrec','SASRec, 5 эпох')]:rows.append([label]+[val(ds,m) for ds in names])
table(rows,[106,131,131,131])
p('Таблица 1. Test NDCG@10, среднее ± выборочное SD по трём seed. MostPopular детерминирован. Числа получены собственными запусками, не перенесены из публикаций.','small')
fig('quality.png','Рисунок 1. Сравнение качества. Отрезки показывают SD по инициализациям, а не доверительный интервал среднего.',height=151)
rows=[['Контраст','Датасет','Δ NDCG@10','95% CI по пользователям']]
for ds in names:
    for mode in ['pad','bos']:
        r=paired[(paired.dataset==ds)&(paired.contrast==mode+'-zero')].iloc[0]
        rows.append([mode.upper()+' − Zero',names[ds],f'{r.delta_ndcg:+.5f}',f'[{r.ci_user_low:+.5f}; {r.ci_user_high:+.5f}]'])
table(rows,[102,127,100,170])
p('Таблица 2. Парный percentile bootstrap по пользователям (2000 повторов) после усреднения их метрик по seed. CI условен на этих трёх обучениях. Для общей неопределённости важен также SD между seed; поправка на множественные сравнения не применялась.','small')
p('PAD даёт положительную среднюю разность с Zero в данном пилоте; размер и устойчивость эффекта зависят от датасета. Даже положительный интервал по пользователям не доказывает превосходство при других рангах, длинах истории и бюджетах. На MovieLens-1M результат PAD близок к простому MostPopular, что ограничивает практическую силу этого наблюдения.')

page();sub('6.2. Длина истории, ресурсы и абляции')
rows=[['Датасет / срез','N','Zero','PAD','BOS']]
for ds in names:
    for split,label in [('short','n<K'),('full','n=K')]:
        r=lookup.loc[(ds,'zero')]
        rows.append([names[ds]+' / '+label,int(r[split+'_n'])]+[f'{lookup.loc[(ds,m)][split+"_ndcg"]:.5f}' for m in ['zero','pad','bos']])
table(rows,[195,50,84,85,85])
p('Таблица 3. Средний NDCG@10 в срезах по длине известной истории на момент предсказания. Малые полные срезы Amazon требуют осторожности: нулевая метрика на них означает отсутствие target в top-10, а не отсутствие вычисления.','small')
rows=[['Датасет','Zero: с / MiB','PAD: с / MiB','BOS: с / MiB','SASRec: с / MiB']]
for ds in names:
    vals=[]
    for m in ['zero','pad','bos','sasrec']:
        r=lookup.loc[(ds,m)];vals.append(f'{r.train_seconds:.2f} / {r.peak_rss_mb:.0f}')
    rows.append([names[ds]]+vals)
table(rows,[125,93,93,93,95])
p('Таблица 4. Среднее время fit и максимум peak RSS по seed. Запуски последовательные, CPU, arm64 macOS, два вычислительных потока. RSS включает интерпретатор, библиотеки, данные и оценку, а не только модель. Fit включает подготовку представления; загрузка данных исключена. Точные platform и версии Python сохраняются в JSON.','small')
rows=[['Контроль','MovieLens-1M','Beauty','Video Games']]
for m,label in [('pad_balanced','PAD / sqrt(q)'),('start','Левый маркер'),('zero_scaled','Zero, scaling=0.2'),('pad_scaled','PAD, scaling=0.2'),('bos_scaled','BOS, scaling=0.2'),('pad_scaled_frequency','PAD, + частота PAD')]:
    vals=[]
    for ds in names:
        r=abl[(abl.dataset==ds)&(abl.model==m)].iloc[0]
        vals.append(f'{r.ndcg:.5f} ± {r.ndcg_sd:.5f}')
    rows.append([label]+vals)
table(rows,[142,119,119,119])
p('Таблица 5. Диагностические абляции, test NDCG@10. Левый маркер меняет обучение, но не передаёт явную границу при инференсе. Нормированный PAD меняет массу служебных наблюдений. Сравнение с таблицей 1 показывает, что результат зависит не только от наличия метки границы: нормировка и изменение задачи аппроксимации тоже существенны.','small')

page();sub('6.3. Что выполнено и что остаётся проверить')
p('Выполнены воспроизводимые прогоны пяти основных моделей на трёх датасетах: по три seed для Zero/PAD/BOS/SASRec и один для MostPopular (39 обучений). Дополнительно выполнены шесть контрольных вариантов на трёх датасетах и трёх seed (54 обучения). Сохранены конфигурации, сырые ранги, метрики, границы разбиения, контрольные суммы данных и исходного кода. Проверки разреженных операций и протокола проходят.')
p('На текущем этапе можно заключить, что выбор кодировки способен менять качество и стоимость LA-SATF в фиксированном режиме. Нельзя утверждать, что эффект объясняется только явной границей истории или что PAD/BOS в целом превосходят настроенный SASRec. Влияние добавленных наблюдений видно и в контролях, где служебная запись действует преимущественно через обучение.')
p('Ограничения: малые ранги и число итераций; только три инициализации; пять эпох SASRec, без проверки сходимости; одна длина K и одно локальное окно; warm-item фильтрация; условный порядок одновременных событий; готовый 5-core Amazon; один target на пользователя. Результаты нельзя непосредственно сопоставлять с числами исходной статьи из-за различий протокола и настроек.')
p('Дальше планируем: (1) выбирать ранги, L, K, scaling и число итераций по validation при одинаковом числе конфигураций для кодировок; (2) довести SASRec до validation-плато и повторить сравнение при согласованном бюджете настройки; (3) увеличить число seed и проверить устойчивость на других временных границах; (4) отдельно отключать служебный вклад при инференсе в одной обученной модели и уравнивать вклад историй после Hankel-преобразования. Текущий test после пилотного анализа не следует использовать как независимое подтверждение новых гипотез.')
sub('Воспроизводимость')
p('Репозиторий проекта: <link href="https://github.com/rangefanimen-blip/satf-padding-bos" color="#126a7c">github.com/rangefanimen-blip/satf-padding-bos</link>. Для воспроизведения текущих таблиц используйте ревизию репозитория, содержащую src/satf_padding, configs и results. Зафиксируйте SHA использованного commit.')
p('После распаковки пакета: <font name="Body">pip install -r requirements.txt -e .</font>; затем <font name="Body">python -m pytest -q</font> и <font name="Body">python -m satf_padding.experiment --output results/reproduced</font>. Для абляций добавьте <font name="Body">--config configs/ablations.json</font> и отдельную папку. Исходные данные автоматически скачиваются у владельцев. Таблицы и рисунки пересоздаются скриптом <font name="Body">scripts/summarize.py</font>. Инструкции, SHA-256 и один самостоятельный Colab-ноутбук включены в репозиторий. Ноутбук содержит исходники и автоматически получает данные.')
sub('Литература')
refs=[
('1','Rendle S., Freudenthaler C., Schmidt-Thieme L. Factorizing Personalized Markov Chains for Next-Basket Recommendation. WWW, 2010.','https://doi.org/10.1145/1772690.1772773'),
('2','Kang W.-C., McAuley J. Self-Attentive Sequential Recommendation. ICDM, 2018.','https://arxiv.org/abs/1808.09781'),
('3','Sun F. et al. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. CIKM, 2019.','https://arxiv.org/abs/1904.06690'),
('4','Frolov E., Oseledets I. Tensor-Based Sequential Learning via Hankel Matrix Representation for Next Item Recommendations. IEEE Access, 11:6357–6371, 2023.','https://doi.org/10.1109/ACCESS.2023.3234863'),
('5','Ji Y., Sun A., Zhang J., Li C. A Critical Study on Data Leakage in Recommender System Offline Evaluation. ACM TOIS, 41(3), 2023.','https://arxiv.org/abs/2010.11060'),
('6','Harper F. M., Konstan J. A. The MovieLens Datasets: History and Context. ACM TiiS, 5(4), 2015.','https://doi.org/10.1145/2827872'),
('7','McAuley Lab. Amazon product data, 2014. Категории Beauty и Video Games, 5-core.','https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html')]
for n,title,url in refs:p(f'[{n}] {title} <link href="{url}" color="#126a7c">Ссылка</link>.','small')

def footer(canvas,doc):
    canvas.setFont('Body',8);canvas.setFillColor(colors.HexColor('#73808a'))
    canvas.drawRightString(A4[0]-48,25,str(doc.page))

doc=SimpleDocTemplate(str(OUT/'report.pdf'),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=39,bottomMargin=40,title='Паддинг и метка начала истории в тензорных последовательных рекомендациях',author='',subject='Промежуточный отчёт',pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'report_source.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Промежуточный отчёт</title><body>'+''.join('<div>'+x.replace('\n','<br>')+'</div>' for x in source)+'</body></html>')
print(OUT/'report.pdf')
