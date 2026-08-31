"""Central configuration, read from environment (.env supported)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- LLM ---
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
# Any of these signals mean the Anthropic SDK can resolve credentials.
ANTHROPIC_KEY_PRESENT = bool(
    os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
)

# --- Vector store ---
# "auto": use MongoDB when MONGODB_URI is set, else the local persisted store.
VECTOR_STORE = os.getenv("VECTOR_STORE", "auto")
MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB = os.getenv("MONGODB_DB", "unicorn_rag")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "chunks")
MONGODB_VECTOR_INDEX = os.getenv("MONGODB_VECTOR_INDEX", "vector_index")

# Largest embedding dimension MongoDB will accept from this app.
#
# This is a data-loss guard, not a tuning knob. TF-IDF produces ~20,000-d
# vectors that are ~99.5% zeros; stored densely in BSON that is ~289 KB per
# chunk. A 2,000-chunk corpus becomes 548 MB, which exhausted an Atlas M0
# (512 MB) and blocked every write on the cluster — and because Atlas
# $vectorSearch indexes are provisioned for a fixed dimension (384 here), the
# oversized vectors silently stopped matching the index too, so retrieval fell
# back to client-side cosine without a word.
#
# That happened for real, triggered by an embedder that failed to load and fell
# back to TF-IDF, after which the startup dimension-heal re-embedded the whole
# corpus into MongoDB. Atlas' own vector search caps at 4096 dimensions, so
# anything above this is a bug rather than a configuration.
MAX_MONGO_EMBEDDING_DIM = int(os.getenv("MAX_MONGO_EMBEDDING_DIM", "4096"))

# --- Embeddings ---
# "auto": voyage > sentence-transformers > tf-idf, whichever is available.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "auto")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL", "BAAI/bge-small-en-v1.5"
)

# --- Chunking (upgraded over the paper's fixed 150-word cuts) ---
CHUNK_TARGET_WORDS = int(os.getenv("CHUNK_TARGET_WORDS", "220"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "40"))

# --- Retrieval ---
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "7"))  # paper found k=7 optimal
RRF_K = 60  # reciprocal-rank-fusion constant
# Depth of each subquery's candidate list before reciprocal-rank fusion.
# Defaults are the measured optimum on data/eval/goldens.json (hit@7 = 0.882);
# raising them made retrieval worse. Re-measure with scripts/evaluate.py before
# changing.
FUSION_POOL_MULTIPLIER = int(os.getenv("FUSION_POOL_MULTIPLIER", "1"))
FUSION_POOL_MIN = int(os.getenv("FUSION_POOL_MIN", "10"))

# --- Cross-encoder reranking ---
# Fusion ranks on topical similarity; a cross-encoder reads question and
# passage together and can tell that a passage *answers* the question. It
# reranks the top RERANK_POOL fused candidates down to k.
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1") not in ("0", "false", "False")
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
RERANK_POOL = int(os.getenv("RERANK_POOL", "40"))
# Rank-fusion constant for blending the fusion and cross-encoder orderings.
# Small values sharpen the difference between top and mid ranks.
RERANK_BLEND_K = int(os.getenv("RERANK_BLEND_K", "5"))

# --- Discovery connectors (YouTube / RSS) ---
# Official YouTube Data API v3 key. Optional booster: non-API strategies are
# always tried first (user preference); the API is a reliable search fallback.
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")


def _csv(name: str, default: str) -> list[str]:
    return [s.strip() for s in os.getenv(name, default).split(",") if s.strip()]


# Public proxy instances (keyless). They fetch YouTube data from *their*
# servers, which sidesteps local IP bot-checks; availability varies, so
# several are tried in order. Override with comma-separated env values.
INVIDIOUS_INSTANCES = _csv(
    "INVIDIOUS_INSTANCES",
    "https://yewtu.be,https://inv.nadeko.net,https://invidious.nerdvpn.de",
)
PIPED_INSTANCES = _csv(
    "PIPED_INSTANCES",
    "https://pipedapi.kavin.rocks,https://pipedapi.adminforge.de",
)
DISCOVERY_MAX_RESULTS = int(os.getenv("DISCOVERY_MAX_RESULTS", "15"))
# A known-captioned video used to test transcript access before a long sweep.
ACCESS_PROBE_VIDEO_ID = os.getenv("ACCESS_PROBE_VIDEO_ID", "-Z1v2vuyYes")
# Optional local ASR for sources without captions (pip install faster-whisper)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
# YouTube sometimes bot-checks datacenter/corporate IPs. Passing browser
# cookies is the documented fix: set one of these (e.g. "chrome", "edge",
# "firefox" — or a path to a Netscape cookies.txt file).
YTDLP_COOKIES_FROM_BROWSER = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "")
# Defaults to secrets/cookies.txt when present, so dropping an exported file
# there is the whole setup. The directory is git-ignored: these are live
# session credentials for your Google account, not config.
_DEFAULT_COOKIE_PATH = ROOT / "secrets" / "cookies.txt"
YTDLP_COOKIES_FILE = os.getenv("YTDLP_COOKIES_FILE", "") or (
    str(_DEFAULT_COOKIE_PATH) if _DEFAULT_COOKIE_PATH.exists() else ""
)

# --- Ingest limits ---
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(10 * 1024 * 1024)))  # 10 MB
MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "50"))
MIN_TEXT_CHARS = int(os.getenv("MIN_TEXT_CHARS", "40"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "2000000"))
MAX_BATCH_DOCS = int(os.getenv("MAX_BATCH_DOCS", "200"))

# --- Grounding gate (abstention) --------------------------------------------
# Retrieval always returns its best k chunks — "best" is a ranking, not a
# judgement that anything answers the question. Without a gate the pipeline
# answers questions the corpus knows nothing about and attaches [S1] citations
# to whatever it retrieved, which reads as evidence. This gate decides whether
# to answer at all.
ABSTAIN_ENABLED = os.getenv("ABSTAIN_ENABLED", "1") not in ("0", "false", "False")

# With credentials the gate is an LLM groundedness check, which generalises.
# Offline it falls back to the thresholds below, combined as:
#     answer if similarity >= MIN_SIMILARITY
#          and (coverage >= MIN_COVERAGE or rerank >= MIN_RERANK)
#
# CALIBRATION, AND ITS LIMITS: these were fitted on 15 answerable and 14
# unanswerable questions against the fictional sample corpus, where they
# separate perfectly. That is 29 points and three thresholds — the fit is
# tighter than the evidence. Treat them as a floor that catches blatant cases,
# not a solved problem, and re-measure on your own corpus with
# scripts/measure_abstention.py. Similarity scale is embedder-specific: BGE
# compresses unrelated text into ~0.5-0.7, TF-IDF puts it near 0.
# Unset by default: the similarity floor has no meaning independent of the
# embedding space, so its default lives on the embedder class
# (`BaseEmbedder.abstain_min_similarity`) — 0.64 for BGE, and 0.0 for TF-IDF,
# where the two distributions were measured to overlap completely. Setting this
# env var overrides whichever embedder is in use, so set it only after running
# scripts/measure_abstention.py against your own corpus.
_min_sim = os.getenv("ABSTAIN_MIN_SIMILARITY")
ABSTAIN_MIN_SIMILARITY = float(_min_sim) if _min_sim else None
ABSTAIN_MIN_COVERAGE = float(os.getenv("ABSTAIN_MIN_COVERAGE", "0.35"))
ABSTAIN_MIN_RERANK = float(os.getenv("ABSTAIN_MIN_RERANK", "-0.7"))

# --- Evaluation / refinement ---
REFINE_HALLUCINATION_THRESHOLD = float(
    os.getenv("REFINE_HALLUCINATION_THRESHOLD", "0.02")
)
# Which golden set to score against, relative to data/eval/ (or an absolute
# path). The default set is labelled against a real swept corpus; point this at
# goldens.samples.json to evaluate the bundled sample corpus instead.
EVAL_GOLDENS_FILE = os.getenv("EVAL_GOLDENS_FILE", "goldens.json")

DATA_DIR = ROOT / "data"
# Overridable so a container can mount the index on a volume, and so a second
# instance can run against a scratch index without touching your real corpus.
INDEX_DIR = Path(os.getenv("INDEX_DIR") or DATA_DIR / "index")
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
# Small fictional corpus shipped with the repo so a fresh clone has something
# to retrieve over before any real media has been collected.
SAMPLES_DIR = DATA_DIR / "samples"
FRONTEND_DIR = ROOT / "frontend"

# Where downloaded embedding/reranker models are cached. Defaults to the
# platform's user cache directory — NOT the working tree, which is what a
# relative path silently produced on Linux and macOS, and not a root-owned
# container path, where the mkdir fails and the engine quietly downgrades to
# TF-IDF. Override for read-only homes or a shared volume.
def _default_model_cache() -> Path:
    if os.name == "nt" and os.getenv("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "UnicornRAG" / "model_cache"
    xdg = os.getenv("XDG_CACHE_HOME")
    return Path(xdg) / "unicorn-rag" if xdg else Path.home() / ".cache" / "unicorn-rag"


MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR") or _default_model_cache())

INDEX_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
