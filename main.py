import os
import uuid
import re

from flask import (
    Flask, render_template, request,
    send_file, redirect, url_for, flash
)
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# ==========================
#        НАСТРОЙКИ
# ==========================

app = Flask(__name__)
app.secret_key = "super-secret-key"
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ускорения
os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_grad_enabled(False)

# ==========================
#       УСТРОЙСТВО
# ==========================

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("🔥 Используем GPU")
else:
    device = torch.device("cpu")
    print("⚙️ GPU нет — работаем на CPU")

# ==========================
#       ЗАГРУЗКА МОДЕЛИ
# ==========================

MODEL_PATH = "model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

LABELS = {
    0: "Отрицательная",
    1: "Нейтральная",
    2: "Положительная"
}

# ==========================
#  АВТОМАТИЧЕСКАЯ НОРМАЛИЗАЦИЯ
# ==========================

def normalize_text(text: str) -> str:
    """
    Простая нормализация текста:
    - приводим к строке
    - убираем лишние пробелы
    - можем добавить доп. шаги (лемматизация и т.п., если понадобится)
    """
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

# ==========================
#     УСКОРЕННЫЙ ИНФЕРЕНС
# ==========================

def predict_batch(texts, batch_size=128):
    """
    Быстрый батчевый инференс:
    - батчи по 128
    - max_length = 128
    - работа на GPU, если есть
    """
    all_preds = []

    print("⌛ Токенизация и инференс...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Model", ncols=80):
        batch = texts[i:i + batch_size]

        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)

    return all_preds

# ==========================
#           ROUTES
# ==========================

@app.route("/")
def index():
    # шаблон index.html уже есть
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Обработка тестового CSV:
    - колонка text (обязательно)
    - src (опционально)
    Результат:
    - полный размеченный CSV
    - submission.csv (id,label)
    - страница с таблицей и визуализациями
    """
    if 'file' not in request.files:
        flash("Файл не найден.")
        return redirect(url_for("index"))

    file = request.files['file']
    if file.filename == "":
        flash("Файл не выбран.")
        return redirect(url_for("index"))

    try:
        df = pd.read_csv(file)
    except Exception as e:
        flash(f"Ошибка чтения: {e}")
        return redirect(url_for("index"))

    if 'text' not in df.columns:
        flash("В CSV должна быть колонка 'text'.")
        return redirect(url_for("index"))

    # Нормализация текста
    df['text_norm'] = df['text'].apply(normalize_text)
    texts = df['text_norm'].astype(str).tolist()

    print(f"📄 Загружено строк: {len(texts)}")
    print("🚀 Запускаем классификацию...")

    preds = predict_batch(texts)

    print("✅ Классификация завершена")

    df['pred'] = preds
    df['label_name'] = df['pred'].map(LABELS)

    # Если есть src — используем, если нет — ставим 'unknown'
    if 'src' not in df.columns:
        df['src'] = 'unknown'

    # Статистика по классам для визуализации
    counts = df['label_name'].value_counts().to_dict()

    # Генерим уникальный ID для файлов
    file_id = str(uuid.uuid4())

    # Полный размеченный CSV
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.csv")
    df.to_csv(full_path, index=False, encoding="utf-8")

    # Submission CSV (формат id,label)
    if 'id' in df.columns:
        sub_df = pd.DataFrame({
            "id": df["id"],
            "label": df["pred"]
        })
    else:
        sub_df = pd.DataFrame({
            "id": range(len(df)),
            "label": df["pred"]
        })

    sub_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_submission.csv")
    sub_df.to_csv(sub_path, index=False, encoding="utf-8")

    # Превью для таблицы
    preview_df = df.head(200)

    return render_template(
        "results.html",
        file_id=file_id,
        table=preview_df.to_dict(orient="records"),
        counts=counts,
        labels_order=list(LABELS.values())
    )


@app.route("/download/<file_id>")
def download(file_id):
    """
    Скачать полный размеченный CSV (text + pred + label_name + src).
    """
    path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.csv")
    if not os.path.exists(path):
        flash("Файл не найден.")
        return redirect(url_for("index"))

    return send_file(
        path,
        as_attachment=True,
        download_name="predicted.csv",
        mimetype="text/csv"
    )


@app.route("/submission/<file_id>")
def submission(file_id):
    """
    Скачать submission.csv в формате id,label.
    """
    sub_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_submission.csv")
    if not os.path.exists(sub_path):
        flash("Submission файл ещё не сформирован.")
        return redirect(url_for("index"))

    return send_file(
        sub_path,
        as_attachment=True,
        download_name="submission.csv",
        mimetype="text/csv"
    )


@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Оценка качества модели по macro-F1:
    - ожидается CSV с колонками text и label (0,1,2)
    """
    if 'file' not in request.files:
        flash("Файл не найден.")
        return redirect(url_for("index"))

    file = request.files['file']
    if file.filename == "":
        flash("Файл не выбран.")
        return redirect(url_for("index"))

    try:
        df = pd.read_csv(file)
    except Exception as e:
        flash(f"Ошибка чтения: {e}")
        return redirect(url_for("index"))

    if 'text' not in df.columns or 'label' not in df.columns:
        flash("В CSV должны быть колонки 'text' и 'label'.")
        return redirect(url_for("index"))

    # Нормализация текста, как и при инференсе
    df['text_norm'] = df['text'].apply(normalize_text)
    texts = df['text_norm'].astype(str).tolist()
    true_labels = df['label'].tolist()

    print(f"📄 Строк для оценки: {len(texts)}")
    print("🚀 Предсказания для метрики...")

    preds = predict_batch(texts)

    macro_f1 = f1_score(true_labels, preds, average="macro")

    report = classification_report(
        true_labels,
        preds,
        target_names=[LABELS[i] for i in sorted(LABELS.keys())],
        digits=4
    )

    return render_template(
        "metrics.html",
        macro_f1=round(macro_f1, 4),
        report=report
    )


if __name__ == "__main__":
    # host="0.0.0.0" — пригодится для деплоя
    app.run(debug=True)
