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


def load_models():
    global _species_processor
    global _species_model
    global _breed_processor
    global _breed_model

    if _species_model is None:
        source = _resolve_source(
            SPECIES_DIR,
            SPECIES_MODEL,
            "species",
        )

        _species_processor = AutoImageProcessor.from_pretrained(source)
        _species_model = AutoModelForImageClassification.from_pretrained(
            source,
            ignore_mismatched_sizes=True,
        )
        _species_model.to(DEVICE)
        _species_model.eval()

    if _breed_model is None:
        source = _resolve_source(
            BREED_DIR,
            BREED_MODEL,
            "breed",
        )

        _breed_processor = AutoImageProcessor.from_pretrained(source)
        _breed_model = AutoModelForImageClassification.from_pretrained(
            source,
            ignore_mismatched_sizes=True,
        )
        _breed_model.to(DEVICE)
        _breed_model.eval()


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

    output = []

    for value, index in zip(values.tolist(), indexes.tolist()):
        label = model.config.id2label.get(index, str(index))

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