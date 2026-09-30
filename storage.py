from pathlib import Path
from uuid import uuid4

from PIL import Image

from config import IMAGE_DIR


ALLOWED = {".jpg", ".jpeg", ".png", ".webp"}


def save_image(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix not in ALLOWED:
        raise ValueError("Formato não suportado.")

    filename = f"{uuid4().hex}{suffix}"
    path = IMAGE_DIR / filename

    image = Image.open(uploaded_file)
    image.verify()

    uploaded_file.seek(0)
    path.write_bytes(uploaded_file.getvalue())

    return filename, path
