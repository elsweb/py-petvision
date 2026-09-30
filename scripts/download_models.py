from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huggingface_hub import snapshot_download

from config import (
    SPECIES_MODEL,
    BREED_MODEL,
    SPECIES_DIR,
    BREED_DIR,
)


def download(model_id, destination):
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print(f"Modelo: {model_id}")
    print(f"Destino: {destination}")
    print("=" * 70)

    snapshot_download(
        repo_id=model_id,
        local_dir=str(destination),
    )


if __name__ == "__main__":
    download(
        SPECIES_MODEL,
        SPECIES_DIR,
    )

    download(
        BREED_MODEL,
        BREED_DIR,
    )

    print()
    print("Modelos baixados.")
