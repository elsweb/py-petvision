from pathlib import Path
import threading

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

from config import (
    SPECIES_DIR,
    BREED_DIR,
    SPECIES_MODEL,
    BREED_MODEL,
    MIN_SPECIES_SAMPLES,
    MIN_BREED_SAMPLES,
)
from database import (
    feedback_rows,
    create_training,
    finish_training,
)
from inference import reload_models


class FeedbackDataset(Dataset):
    def __init__(self, rows, processor, labels):
        self.rows = rows
        self.processor = processor
        self.labels = labels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(row["path"]).convert("RGB")
        encoded = self.processor(images=image, return_tensors="pt")

        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": self.labels[row["label"]],
        }


def _layer_data(layer):
    rows = feedback_rows()
    if layer == "species":
        return [{**row, "label": row["species"]} for row in rows if row["species"]]
    return [{**row, "label": row["breed"]} for row in rows if row["breed"]]


def _resolve_source(custom_dir, base_dir, model_id):
    if (custom_dir / "config.json").exists():
        return str(custom_dir)
    if (Path(base_dir) / "config.json").exists():
        return str(base_dir)
    return model_id


def _merge_labels(base_id2label, feedback_labels):
    id2label = {int(k): v for k, v in base_id2label.items()}
    label2id = {v: int(k) for k, v in id2label.items()}

    new_labels = []
    next_id = max(id2label.keys(), default=-1) + 1

    for label in feedback_labels:
        if label not in label2id:
            id2label[next_id] = label
            label2id[label] = next_id
            new_labels.append(label)
            next_id += 1

    return id2label, label2id, new_labels


def _expand_classifier(model, old_size, new_size):
    # Em modelos ViT, o classificador pode se chamar 'classifier' ou fazer parte de 'heads'
    if hasattr(model, "classifier"):
        head = model.classifier
        new_head = torch.nn.Linear(head.in_features, new_size)
        with torch.no_grad():
            new_head.weight[:old_size] = head.weight[:old_size]
            new_head.bias[:old_size] = head.bias[:old_size]
        model.classifier = new_head
    else:
        # Fallback para modelos que usam 'heads'
        head = model.heads.head if hasattr(model, "heads") else model.classifier
        # Cria a nova camada baseada no modelo original
        in_features = model.config.hidden_size
        new_head = torch.nn.Linear(in_features, new_size)
        with torch.no_grad():
             new_head.weight[:old_size] = head.weight[:old_size]
             new_head.bias[:old_size] = head.bias[:old_size]
        # Aplica no modelo (depende da arquitetura exata do ViT)
        if hasattr(model, "heads"):
            model.heads.head = new_head
        else:
            model.classifier = new_head


def train_layer(
    layer,
    epochs=2,
    batch_size=4,
    learning_rate=2e-5, 
    progress_callback=None,
):
    rows = _layer_data(layer)
    minimum = MIN_SPECIES_SAMPLES if layer == "species" else MIN_BREED_SAMPLES

    if len(rows) < minimum:
        raise ValueError(f"São necessários pelo menos {minimum} exemplos. Existem {len(rows)}.")

    feedback_labels = sorted({row["label"] for row in rows})
    if not feedback_labels:
        raise ValueError("Nenhuma classe válida foi encontrada.")

    if layer == "species":
        base_dir, model_id = SPECIES_DIR, SPECIES_MODEL
    else:
        base_dir, model_id = BREED_DIR, BREED_MODEL

    custom_dir = Path(base_dir).parent / "custom" / layer
    source = _resolve_source(custom_dir, base_dir, model_id)

    processor = AutoImageProcessor.from_pretrained(source)
    model = AutoModelForImageClassification.from_pretrained(source)

    base_id2label = dict(model.config.id2label)
    base_size = len(base_id2label)
    merged_id2label, merged_label2id, added = _merge_labels(base_id2label, feedback_labels)

    if added:
        _expand_classifier(model, base_size, len(merged_id2label))

    model.config.id2label = merged_id2label
    model.config.label2id = merged_label2id
    model.config.num_labels = len(merged_id2label)
    label_map = merged_label2id

    # Preparar os dados para o Split Estratificado
    dataset = FeedbackDataset(rows, processor, label_map)
    targets = [label_map[row["label"]] for row in rows]

    # Se tiver poucas imagens por classe, o split tradicional falha. Vamos garantir 1 de cada na validação.
    if len(np.unique(targets)) < 2:
         raise ValueError("Você precisa de pelo menos 2 raças diferentes no banco para treinar.")

    # StratifiedShuffleSplit garante que todas as raças apareçam no treino e na validação
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    # Se o dataset for muito pequeno (ex: 5 imagens), pode ser que falte dados para o split.
    # O max(1) garante que sempre terá pelo menos 1 imagem na validação.
    try:
        train_idx, val_idx = next(splitter.split(targets, targets))
    except ValueError:
        # Fallback se não houver amostras suficientes por classe
        raise ValueError("Você precisa de mais imagens de cada raça para fazer o treino e a validação.")

    train_dataset = Subset(dataset, train_idx)
    validation_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # IMPORTANTE: Descongelar o modelo inteiro para ele aprender a nova raça
    for param in model.parameters():
        param.requires_grad = True

    # Learning rate um pouco menor para não destruir o modelo antigo (evitar Catastrophic Forgetting)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    total_steps = max(1, epochs * len(train_loader))
    step = 0
    last_loss = 0.0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels_tensor = batch["labels"].to(device)

            output = model(pixel_values=pixel_values, labels=labels_tensor)
            loss = output.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            last_loss = float(loss.detach().cpu())
            step += 1
            if progress_callback:
                progress_callback(int(step / total_steps * 85))

    model.eval()

    y_true = []
    y_pred = []

    with torch.inference_mode():
        for batch in validation_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels_tensor = batch["labels"].to(device)

            output = model(pixel_values=pixel_values, labels=labels_tensor)
            logits = output.logits
            last_loss = float(output.loss.detach().cpu())
            predictions = logits.argmax(dim=-1)

            y_true.extend(labels_tensor.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = (correct / total) if total > 0 else 0.0

    labels_presentes = sorted(list(set(y_true + y_pred)))
    target_names = [merged_id2label[i] for i in labels_presentes]
    
    report = classification_report(
        y_true, 
        y_pred, 
        labels=labels_presentes,
        target_names=target_names,
        output_dict=True, 
        zero_division=0
    )

    custom_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(custom_dir)
    processor.save_pretrained(custom_dir)

    if progress_callback:
        progress_callback(100)

    return {
        "samples": len(rows),
        "labels": feedback_labels,
        "new_labels": added,
        "total_classes": len(merged_id2label),
        "loss": last_loss,
        "accuracy": accuracy,
        "device": device,
        "frozen_backbone": False, # Agora sempre False
        "report": report,
    }


class Trainer:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.progress = 0
        self.layer = None
        self.message = "Pronto"
        self.result = None

    def state(self):
        with self.lock:
            return {
                "running": self.running,
                "progress": self.progress,
                "layer": self.layer,
                "message": self.message,
                "result": self.result,
            }

    def start(self, layer, epochs):
        with self.lock:
            if self.running:
                raise RuntimeError("Já existe um treinamento.")

            self.running = True
            self.progress = 1
            self.layer = layer
            self.message = "Preparando dataset..."
            self.result = None

        thread = threading.Thread(
            target=self._worker,
            args=(layer, epochs),
            daemon=True,
        )
        thread.start()

    def _worker(self, layer, epochs):
        rows = _layer_data(layer)
        run_id = create_training(layer, len(rows), epochs)

        try:
            with self.lock:
                self.message = "Treinando Vision Transformer..."

            def progress(value):
                with self.lock:
                    self.progress = value

            result = train_layer(
                layer=layer,
                epochs=epochs,
                progress_callback=progress,
            )

            finish_training(
                run_id,
                "completed",
                loss=result["loss"],
                accuracy=result["accuracy"],
                report=result["report"],
            )

            reload_models()

            with self.lock:
                self.running = False
                self.progress = 100
                self.message = "Treinamento concluído."
                self.result = result

        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            finish_training(run_id, "failed", error=tb)
            with self.lock:
                self.running = False
                self.progress = 0
                self.message = f"{exc}\n\n{tb}"
                self.result = None


trainer = Trainer()