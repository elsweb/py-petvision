from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
MODEL_DIR = ROOT / "models"

SPECIES_DIR = MODEL_DIR / "species"
BREED_DIR = MODEL_DIR / "breed"

DB_PATH = DATA_DIR / "petvision.db"

SPECIES_MODEL = os.getenv(
    "PETVISION_SPECIES_MODEL",
    "akahana/vit-base-cats-vs-dogs",
)

BREED_MODEL = os.getenv(
    "PETVISION_BREED_MODEL",
    "MaxPowerUnlimited/vit-base-oxford-iiit-pets",
)

MIN_SPECIES_SAMPLES = 20
MIN_BREED_SAMPLES = 5

for directory in [
    DATA_DIR,
    IMAGE_DIR,
    MODEL_DIR,
    SPECIES_DIR,
    BREED_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
