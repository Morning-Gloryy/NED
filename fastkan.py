import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from transformers import AutoTokenizer, AutoModel
from underthesea import word_tokenize, text_normalize
import numpy as np
import unicodedata
import re
import os
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
# --- Import KAN ---  https://github.com/ZiyaoLi/fast-kan 
from fastkan import FastKAN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Đang sử dụng thiết bị: {DEVICE}")

DATA_FILE_PATH = 'content/dataset.csv'
RESULTS_DIR = 'training_results_clc_optimized'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

BATCH_SIZE = 32
EPOCHS = 500 
EARLY_STOPPING_PATIENCE = 7
SCHEDULER_PATIENCE = 3 

TFIDF_MAX_FEATURES = 2000
PHOBERT_MODEL_NAME = "vinai/phobert-base-v2"

def load_stopwords(filepath='lib/vietnamese-stopwords.txt'):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return set(f.read().splitlines())
    except FileNotFoundError:
        print(f"Cảnh báo: Không tìm thấy tệp stopwords tại '{filepath}'.")
        return set()

def preprocess_text(text, stopwords_set):
    text = str(text).lower()
    text = unicodedata.normalize('NFC', text)
    text = text_normalize(text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords_set]
    return ' '.join(words)

def load_and_preprocess_data(file_path):
    print(f"\n[Bước 1] Đang tải và tiền xử lý dữ liệu từ '{file_path}'...")
    df = pd.read_csv(file_path, sep=';')
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV phải có cột 'text' và 'label'.")

    stopwords_set = load_stopwords()
    tqdm.pandas(desc="Đang tiền xử lý văn bản")
    df['processed_text'] = df['text'].progress_apply(lambda x: preprocess_text(x, stopwords_set))
    
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    
    print("Hoàn thành tiền xử lý.")
    return df, label_encoder

def extract_tfidf_features(df, max_features):
    print(f"\n[Bước 2a] Đang trích xuất đặc trưng TF-IDF (max_features={max_features})...")
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tfidf = vectorizer.fit_transform(df['processed_text']).toarray()
    print(f"Kích thước ma trận TF-IDF: {X_tfidf.shape}")
    return X_tfidf, vectorizer

def extract_phobert_embeddings(df, model_name, batch_size):
    print(f"\n[Bước 2b] Đang trích xuất đặc trưng nhúng từ PhoBERT...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()

    embeddings = []
    text_list = df['processed_text'].tolist()
    
    for i in tqdm(range(0, len(text_list), batch_size), desc="Trích xuất PhoBERT"):
        batch_texts = text_list[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.pooler_output
        embeddings.append(batch_embeddings.cpu().numpy())

    X_phobert = np.vstack(embeddings)
    print(f"Kích thước ma trận PhoBERT embeddings: {X_phobert.shape}")
    del model, tokenizer
    torch.cuda.empty_cache()
    return X_phobert

class KANClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, kan_params={}):
        super(KANClassifier, self).__init__()
        default_params = {'grid_min': -2.0, 'grid_max': 2.0, 'num_grids': 8}
        self.kan_params = {**default_params, **kan_params}
        kan_layers = [input_dim] + hidden_dims + [num_classes]
        self.kan = FastKAN(layers_hidden=kan_layers, **self.kan_params)

    def forward(self, x):
        return self.kan(x)

class LinearHeadKAN(nn.Module):
    def __init__(self, input_dim, reduced_dim, hidden_dims, num_classes, kan_params={}):
        super(LinearHeadKAN, self).__init__()
        self.linear_head = nn.Linear(input_dim, reduced_dim)
        self.activation = nn.LeakyReLU(0.1)
        kan_input_dim = reduced_dim
        kan_layers = [kan_input_dim] + hidden_dims + [num_classes]
        default_params = {'grid_min': -2.0, 'grid_max': 2.0, 'num_grids': 8}
        self.kan_params = {**default_params, **kan_params}
        self.kan = FastKAN(layers_hidden=kan_layers, **self.kan_params)
        
    def forward(self, x):
        x = self.linear_head(x)
        x = self.activation(x)
        return self.kan(x)

class EarlyStopping:
    """Dừng quá trình huấn luyện sớm nếu validation loss không cải thiện sau một số epochs nhất định."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Lưu mô hình khi validation loss giảm."""
        if self.verbose:
            self.trace_func(f'Validation loss giảm ({self.val_loss_min:.6f} --> {val_loss:.6f}). Đang lưu mô hình ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_evaluate_kan(model, X_train, y_train, X_val, y_val, X_test, y_test, 
                       epochs, lr, batch_size, weight_decay, model_save_path):
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=SCHEDULER_PATIENCE)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True, path=model_save_path)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in pbar_train:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            pbar_train.set_postfix(loss=loss.item(), acc=train_correct/train_total)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)

        # Đánh giá trên tập validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for inputs, labels in pbar_val:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                pbar_val.set_postfix(loss=loss.item(), acc=val_correct/val_total)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)

        print(f"Epoch {epoch+1}/{epochs} Summary | "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    # Tải mô hình tốt nhất đã được lưu bởi EarlyStopping và đánh giá trên tập test
    print(f"Đang tải mô hình tốt nhất từ '{model_save_path}' để đánh giá cuối cùng.")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return all_preds, all_labels, history

def plot_history(history, file_path):
    plt.figure(figsize=(12, 5))
    # Giới hạn trục x theo số epochs thực sự đã chạy
    epochs_ran = len(history['train_acc'])
    x_axis = range(1, epochs_ran + 1)
    
    plt.subplot(1, 2, 1)
    plt.plot(x_axis, history['train_acc'], label='Train Accuracy')
    plt.plot(x_axis, history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x_axis, history['train_loss'], label='Train Loss')
    plt.plot(x_axis, history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, file_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(file_path)
    plt.close()


def main():
    df, label_encoder = load_and_preprocess_data(DATA_FILE_PATH)
    y = df['label_encoded'].values
    class_names = label_encoder.classes_
    num_classes = len(class_names)

    X_tfidf, _ = extract_tfidf_features(df, max_features=TFIDF_MAX_FEATURES)
    X_phobert = extract_phobert_embeddings(df, PHOBERT_MODEL_NAME, BATCH_SIZE)

    kan_experiments = {
        "KAN_LinearHead_Deeper_TFIDF": {
            "model_class": LinearHeadKAN,
            "params": {"reduced_dim": 256, "hidden_dims": [128, 64], "num_classes": num_classes},
            "features": "TFIDF",
            "lr": 1e-3, 
            "weight_decay": 1e-5
        },
        "KAN_LinearHead_Simple_TFIDF": {
            "model_class": LinearHeadKAN,
            "params": {"reduced_dim": 128, "hidden_dims": [64], "num_classes": num_classes},
            "features": "TFIDF",
            "lr": 1e-3,
            "weight_decay": 1e-5
        },
        "KAN_Simple_PhoBERT": {
            "model_class": KANClassifier,
            "params": {"hidden_dims": [128, 64], "num_classes": num_classes},
            "features": "PhoBERT",
            "lr": 5e-5, 
            "weight_decay": 1e-4 
        },
        "KAN_Wide_PhoBERT": {
            "model_class": KANClassifier,
            "params": {"hidden_dims": [256], "num_classes": num_classes},
            "features": "PhoBERT",
            "lr": 5e-5,
            "weight_decay": 1e-4
        }
    }
    
    all_final_results = {}

    for exp_name, config in kan_experiments.items():
        print(f"\n{'='*30}\n[Thí nghiệm] Bắt đầu: {exp_name}\n{'='*30}")

        if config["features"] == "TFIDF":
            X = X_tfidf
        elif config["features"] == "PhoBERT":
            X = X_phobert
        else:
            continue
        
        input_dim = X.shape[1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

        exp_dir = os.path.join(RESULTS_DIR, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        model_save_path = os.path.join(exp_dir, f"{exp_name}_best.pth")

        ModelClass = config["model_class"]
        model_params = config["params"].copy()
        model_params['input_dim'] = input_dim
        model = ModelClass(**model_params)
        print(f"Đã tạo mô hình: {ModelClass.__name__} với params: {model_params}")
        print(f"Tham số huấn luyện - LR: {config['lr']}, Weight Decay: {config['weight_decay']}")
        
        y_pred, y_true, history = train_evaluate_kan(
            model, X_train, y_train, X_val, y_val, X_test, y_test,
            epochs=EPOCHS, 
            lr=config['lr'],
            batch_size=BATCH_SIZE,
            weight_decay=config['weight_decay'], 
            model_save_path=model_save_path
        )
        
        print(f"\nĐang lưu kết quả cho thí nghiệm: {exp_name}")
        plot_history(history, os.path.join(exp_dir, "train_history.png"))
        plot_confusion_matrix(y_true, y_pred, class_names, os.path.join(exp_dir, "confusion_matrix.png"))
        
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        all_final_results[exp_name] = report
        
        with open(os.path.join(exp_dir, 'classification_report.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        
        print(f"Đã lưu kết quả vào thư mục: {exp_dir}")
        del model
        torch.cuda.empty_cache()

    summary_path = os.path.join(RESULTS_DIR, 'all_experiments_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_final_results, f, ensure_ascii=False, indent=4)
    print(f"\n[HOÀN TẤT] Báo cáo tổng hợp đã được lưu tại: {summary_path}")

if __name__ == '__main__':
    main()