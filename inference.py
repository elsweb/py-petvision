from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from config import (
    SPECIES_DIR,
    BREED_DIR,
    SPECIES_MODEL,
    BREED_MODEL,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_species_processor = None
_species_model = None

_breed_processor = None
_breed_model = None


def _resolve_source(base_dir, model_id, layer):
    """
    Ordem de prioridade do checkpoint de inferência:

    1) models/custom/{layer}  -> fine-tuning local (feedback humano)
    2) base_dir               -> modelo base salvo localmente
    3) model_id               -> modelo base do HuggingFace
    """
    custom_dir = Path(base_dir).parent / "custom" / layer

    if (custom_dir / "config.json").exists():
        return str(custom_dir)

    if (Path(base_dir) / "config.json").exists():
        return str(base_dir)

    return model_id


def _normalize_config(model):
    """
    Garante id2label/label2id com tipos consistentes (int -> str).

    Sem isso, após save_pretrained/from_pretrained as chaves de id2label
    voltam como string ("0", "1", ...) e o lookup por índice int falha,
    fazendo a inferência retornar "0", "1", "2" em vez dos labels reais.
    """
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: int(k) for k, v in id2label.items()}

    model.config.id2label = id2label
    model.config.label2id = label2id
    model.config.num_labels = len(id2label)

    return model


def _load_layer(base_dir, model_id, layer):
    source = _resolve_source(base_dir, model_id, layer)

    processor = AutoImageProcessor.from_pretrained(source)
    model = AutoModelForImageClassification.from_pretrained(
        source,
        ignore_mismatched_sizes=True,
    )
    model.to(DEVICE)
    model.eval()
    model = _normalize_config(model)

    return processor, model


def load_models():
    global _species_processor
    global _species_model
    global _breed_processor
    global _breed_model

    if _species_model is None:
        _species_processor, _species_model = _load_layer(
            SPECIES_DIR,
            SPECIES_MODEL,
            "species",
        )

    if _breed_model is None:
        _breed_processor, _breed_model = _load_layer(
            BREED_DIR,
            BREED_MODEL,
            "breed",
        )


def reload_models():
    global _species_processor
    global _species_model
    global _breed_processor
    global _breed_model

    _species_processor = None
    _species_model = None
    _breed_processor = None
    _breed_model = None

    load_models()


def classify(image, processor, model, top_k=5):
    encoded = processor(
        images=image,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    values, indexes = torch.topk(
        probabilities,
        min(top_k, probabilities.shape[-1]),
    )

    # id2label já normalizado (chaves int) em _load_layer
    id2label = model.config.id2label

    output = []

    for value, index in zip(values.tolist(), indexes.tolist()):
        label = id2label.get(int(index), str(index))

        output.append({
            "label": label,
            "probability": float(value),
        })

    return output


def detect(path):
    load_models()

    image = Image.open(path).convert("RGB")

    species = classify(
        image,
        _species_processor,
        _species_model,
        top_k=2,
    )

    breed = classify(
        image,
        _breed_processor,
        _breed_model,
        top_k=5,
    )

    return {
        "device": DEVICE,
        "species": species,
        "species_top": species[0],
        "breed": breed,
        "breed_top": breed[0],
    }