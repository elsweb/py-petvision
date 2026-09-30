import sqlite3
from datetime import datetime, timezone

from config import DB_PATH


def now():
    return datetime.now(timezone.utc).isoformat()


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = connect()

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            path TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            species TEXT,
            species_probability REAL,
            breed TEXT,
            breed_probability REAL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            species TEXT NOT NULL,
            breed TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            layer TEXT NOT NULL,
            status TEXT NOT NULL,
            samples INTEGER,
            epochs INTEGER,
            loss REAL,
            accuracy REAL,
            created_at TEXT NOT NULL,
            finished_at TEXT,
            error TEXT
        );
    """)

    conn.commit()
    conn.close()


def add_image(filename, path):
    conn = connect()
    cur = conn.execute(
        "INSERT INTO images(filename,path,created_at) VALUES(?,?,?)",
        (filename, str(path), now()),
    )
    conn.commit()
    image_id = cur.lastrowid
    conn.close()
    return image_id


def add_prediction(image_id, result):
    conn = connect()
    conn.execute(
        """
        INSERT INTO predictions(
            image_id,species,species_probability,
            breed,breed_probability,created_at
        ) VALUES(?,?,?,?,?,?)
        """,
        (
            image_id,
            result["species_top"]["label"],
            result["species_top"]["probability"],
            result["breed_top"]["label"],
            result["breed_top"]["probability"],
            now(),
        ),
    )
    conn.commit()
    conn.close()


def add_feedback(image_id, species, breed):
    conn = connect()
    conn.execute(
        """
        INSERT INTO feedback(image_id,species,breed,created_at)
        VALUES(?,?,?,?)
        """,
        (image_id, species, breed, now()),
    )
    conn.commit()
    conn.close()


def feedback_rows():
    conn = connect()
    rows = conn.execute(
        """
        SELECT f.*, i.path, i.filename
        FROM feedback f
        JOIN images i ON i.id=f.image_id
        ORDER BY f.id
        """
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def stats():
    conn = connect()

    total = conn.execute(
        "SELECT COUNT(*) FROM feedback"
    ).fetchone()[0]

    species = conn.execute(
        """
        SELECT species, COUNT(*) count
        FROM feedback
        GROUP BY species
        """
    ).fetchall()

    breeds = conn.execute(
        """
        SELECT breed, COUNT(*) count
        FROM feedback
        WHERE breed IS NOT NULL
        GROUP BY breed
        """
    ).fetchall()

    runs = conn.execute(
        """
        SELECT *
        FROM training_runs
        ORDER BY id DESC
        LIMIT 10
        """
    ).fetchall()

    conn.close()

    return {
        "total": total,
        "species": [dict(x) for x in species],
        "breeds": [dict(x) for x in breeds],
        "runs": [dict(x) for x in runs],
    }


def create_training(layer, samples, epochs):
    conn = connect()
    cur = conn.execute(
        """
        INSERT INTO training_runs(
            layer,status,samples,epochs,created_at
        ) VALUES(?,?,?,?,?)
        """,
        (layer, "running", samples, epochs, now()),
    )
    conn.commit()
    run_id = cur.lastrowid
    conn.close()
    return run_id


def finish_training(run_id, status, loss=None, accuracy=None, error=None):
    conn = connect()
    conn.execute(
        """
        UPDATE training_runs
        SET status=?,loss=?,accuracy=?,finished_at=?,error=?
        WHERE id=?
        """,
        (status, loss, accuracy, now(), error, run_id),
    )
    conn.commit()
    conn.close()
