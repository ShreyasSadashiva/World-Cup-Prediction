"""
Firebase Firestore client singleton.

Credentials are resolved in order:
  1. FIREBASE_CREDENTIALS_PATH  — path to a service account JSON file
  2. FIREBASE_CREDENTIALS_JSON  — full JSON content as an env variable
  3. Application Default Credentials (e.g. if running on GCP)
"""

import json
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def get_client():
    import firebase_admin
    from firebase_admin import credentials, firestore

    if not firebase_admin._apps:
        cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH")
        cred_json = os.environ.get("FIREBASE_CREDENTIALS_JSON")

        if cred_path:
            cred = credentials.Certificate(cred_path)
        elif cred_json:
            cred = credentials.Certificate(json.loads(cred_json))
        else:
            cred = credentials.ApplicationDefault()

        firebase_admin.initialize_app(cred)

    from firebase_admin import firestore as fs
    return fs.client()


def slugify(name: str) -> str:
    """Convert a team name to a safe Firestore document ID."""
    import re
    import unicodedata
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")
