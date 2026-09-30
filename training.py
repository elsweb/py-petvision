from pathlib import Path
import threading

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoImageProcessor, AutoModelForImageClassification

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

        encoded = self.processor(
            images=image,
            return_tensors="pt",
        )

        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": self.labels[row["label"]],
        }


def _layer_data(layer):
    rows = feedback_rows()

    if layer == "species":
        return [
            {
                **row,
                "label": row["species"],
            }
            for row in rows
            if row["species"]
        ]

    return [
        {
            **row,
            "label": row["breed"],
        }
        for row in rows
        if row["breed"]
    ]


def _resolve_source(custom_dir, base_dir, model_id):
    if (custom_dir / "config.json").exists():
        return str(custom_dir)
    if (Path(base_dir) / "config.json").exists():
        return str(base_dir)
    return model_id


def _merge_labels(base_id2label, feedback_labels):
    """
    Mescla o label map do modelo base com os labels vindos do feedback.

    - Labels já conhecidos mantêm o índice original (pesos preservados).
    - Labels novos são anexados no fim (pesos inicializados do zero).

    Retorna:
        id2label   -> dict {índice: label}
        label2id   -> dict {label: índice}
        new_labels -> lista de labels efetivamente adicionados
    """
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
    """
    Expande o head classificador preservando os pesos das classes antigas.
    As novas linhas ficam com a inicialização padrão do PyTorch.
    """
    head = model.classifier

    new_head = torch.nn.Linear(
        head.in_features,
        new_size,
    )

    with torch.no_grad():
        new_head.weight[:old_size] = head.weight[:old_size]
        new_head.bias[:old_size] = head.bias[:old_size]

    model.classifier = new_head


def train_layer(
    layer,
    epochs=2,
    batch_size=4,
    learning_rate=2e-5,
    progress_callback=None,
):
    rows = _layer_data(layer)

    minimum = (
        MIN_SPECIES_SAMPLES
        if layer == "species"
        else MIN_BREED_SAMPLES
    )

    if len(rows) < minimum:
        raise ValueError(
            f"São necessários pelo menos {minimum} exemplos. "
            f"Existem {len(rows)}."
        )

    feedback_labels = sorted({row["label"] for row in rows})

    if not feedback_labels:
        raise ValueError(
            "Nenhuma classe válida foi encontrada."
        )

    if layer == "species":
        base_dir = SPECIES_DIR
        model_id = SPECIES_MODEL
    else:
        base_dir = BREED_DIR
        model_id = BREED_MODEL

    custom_dir = Path(base_dir).parent / "custom" / layer

    source = _resolve_source(custom_dir, base_dir, model_id)

    processor = AutoImageProcessor.from_pretrained(source)

    model = AutoModelForImageClassification.from_pretrained(source)

    # -----------------------------------------------------------------
    # Mergulhar labels: mantém o que o modelo já sabe + adiciona novo
    # -----------------------------------------------------------------

    base_id2label = dict(model.config.id2label)
    base_size = len(base_id2label)

    merged_id2label, merged_label2id, added = _merge_labels(
        base_id2label,
        feedback_labels,
    )

    if added:
        _expand_classifier(model, base_size, len(merged_id2label))

    model.config.id2label = merged_id2label
    model.config.label2id = merged_label2id
    model.config.num_labels = len(merged_id2label)

    label_map = merged_label2id

    # -----------------------------------------------------------------
    # Dataset / split
    # -----------------------------------------------------------------

    dataset = FeedbackDataset(rows, processor, label_map)

    if len(dataset) == 1:
        train_dataset = dataset
        validation_dataset = None
    else:
        validation_count = max(1, int(len(dataset) * 0.2))
        train_count = len(dataset) - validation_count

        if train_count < 1:
            train_count = 1
            validation_count = len(dataset) - 1

        train_dataset, validation_dataset = random_split(
            dataset,
            [train_count, validation_count],
            generator=torch.Generator().manual_seed(42),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = None

    if validation_dataset is not None:
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # -----------------------------------------------------------------
    # Estratégia de treino
    #
    # - Se adicionamos classes novas -> congela backbone.
    #   Só o classificador é atualizado. Preserva o que o modelo base
    #   já sabia e aprende rapidamente o head novo.
    #
    # - Se só estamos refinando classes existentes com poucos dados
    #   -> congela backbone também.
    #
    # - Se temos bastante dado por classe -> fine-tuning completo.
    # -----------------------------------------------------------------

    samples_per_class = len(rows) / max(1, len(feedback_labels))
    small_dataset = samples_per_class < 10

    freeze_backbone = added or small_dataset

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

        trainable = [p for p in model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable,
            lr=1e-3,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
        )

    total_steps = max(1, epochs * len(train_loader))

    step = 0
    last_loss = 0.0

    for epoch in range(epochs):
        model.train()

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels_tensor = batch["labels"].to(device)

            output = model(
                pixel_values=pixel_values,
                labels=labels_tensor,
            )

            loss = output.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            last_loss = float(loss.detach().cpu())
            step += 1

            if progress_callback:
                progress_callback(int(step / total_steps * 85))

    model.eval()

    correct = 0
    total = 0

    if validation_loader is not None:
        with torch.inference_mode():
            for batch in validation_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels_tensor = batch["labels"].to(device)

                logits = model(pixel_values=pixel_values).logits
                predictions = logits.argmax(dim=-1)

                correct += int((predictions == labels_tensor).sum())
                total += int(labels_tensor.numel())

    accuracy = (correct / total) if total > 0 else 0.0

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
        "frozen_backbone": freeze_backbone,
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

        run_id = create_training(
            layer,
            len(rows),
            epochs,
        )

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
            )

            reload_models()

            with self.lock:
                self.running = False
                self.progress = 100
                self.message = "Treinamento concluído."
                self.result = result

        except Exception as exc:
            finish_training(
                run_id,
                "failed",
                error=str(exc),
            )

            with self.lock:
                self.running = False
                self.progress = 0
                self.message = str(exc)
                self.result = None


trainer = Trainer()