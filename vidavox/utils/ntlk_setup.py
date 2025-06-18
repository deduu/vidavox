# utils/nltk_setup.py
from pathlib import Path
import nltk, logging, os

# ── 1.  Figure out the project root (folder that contains main.py) ──────────────
ROOT_DIR = Path(__file__).resolve().parent.parent   # adjust “..” if needed

# ── 2.  Point NLTK to <root>/nltk_data ─────────────────────────────────────────
NLTK_DATA_DIR = ROOT_DIR / "nltk_data"
NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)    # create if missing
nltk.data.path.append(str(NLTK_DATA_DIR))

# ── 3.  Download any missing resources once at start-up ────────────────────────
_REQUIRED = ["punkt_tab", "stopwords", "wordnet", "omw-1.4"]

def ensure_nltk_resources():
    for res in _REQUIRED:
        try:
            nltk.data.find(res)
        except LookupError:
            logging.info(f"Downloading {res} ->{NLTK_DATA_DIR}")
            nltk.download(res, download_dir=str(NLTK_DATA_DIR), quiet=True)


# Call it once to download all the resources
# from app.utils.nltk_setup import ensure_nltk_resources
# ...
# async def lifespan(app: FastAPI):
#     ensure_nltk_resources()     # happens once, before workers start
#     ...
# utils/nltk_bootstrap.py
# import nltk
# from nltk import data as _data
# import logging

# logger = logging.getLogger(__name__)

# def ensure_nltk_resources() -> None:
#     for res, path in (
#         ("punkt", "tokenizers/punkt"),
#         ("stopwords", "corpora/stopwords"),
#         ("wordnet", "corpora/wordnet"),
#     ):
#         try:
#             _data.find(path)
#         except LookupError:
#             logger.info("Downloading NLTK resource %s …", res)
#             nltk.download(res, quiet=True)