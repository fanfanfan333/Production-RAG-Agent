"""
One-off migration script: delete the primary Qdrant collection (currently
holding 3072-dim Gemini vectors) and recreate it empty with the new
dimension defined by settings.EMBEDDING_DIMENSION (1024, BGE).

This bypasses collection_service.delete_collection()'s guard on purpose —
that guard protects the API from accidental deletion, but this script IS
the intentional, one-time migration step.

WHAT THIS DOES NOT TOUCH:
    PostgreSQL `documents` table rows are left untouched. Their vectors
    are gone after this runs, so any document previously marked as
    "processed" will no longer be retrievable until it is re-embedded.
    You will need to re-trigger processing (or re-upload) every document
    afterwards for search to work again.

Usage:
    cd backend
    python scripts/reset_qdrant_collection.py
"""

import asyncio
import sys
from pathlib import Path

# Fix: running `python scripts/reset_qdrant_collection.py` only puts the
# script's own directory (backend/scripts/) on sys.path, NOT backend/.
# Without this, `import app...` can resolve to an unrelated third-party
# package literally named "app" in site-packages instead of the local
# backend/app package. Force backend/ to the front of sys.path first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.db.qdrant import get_qdrant_client
from app.services.vector_service import ensure_collection
from app.utils.logging import get_logger

logger = get_logger(__name__)


async def main() -> None:
    settings = get_settings()
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION

    existing = await client.get_collections()
    names = {c.name for c in existing.collections}

    if collection_name in names:
        info = await client.get_collection(collection_name=collection_name)
        try:
            current_dim = info.config.params.vectors.size
        except Exception:
            current_dim = "unknown"

        print(
            f"Collection '{collection_name}' exists (current dim={current_dim}, "
            f"target dim={settings.EMBEDDING_DIMENSION}).\n"
            f"This will PERMANENTLY DELETE ALL VECTORS in this collection."
        )
        confirm = input("Type 'yes' to continue: ")
        if confirm.strip().lower() != "yes":
            print("Aborted — nothing was deleted.")
            return

        await client.delete_collection(collection_name=collection_name)
        logger.info("Deleted existing collection '%s'", collection_name)
    else:
        logger.info(
            "Collection '%s' does not exist yet — nothing to delete.",
            collection_name,
        )

    await ensure_collection()
    logger.info(
        "Recreated collection '%s' with dim=%d (BGE, COSINE distance)",
        collection_name,
        settings.EMBEDDING_DIMENSION,
    )
    print(
        f"\nDone. Collection '{collection_name}' now expects "
        f"{settings.EMBEDDING_DIMENSION}-dim vectors.\n"
        f"Next step: re-process / re-upload your documents so they get "
        f"re-embedded with the new BGE model."
    )


if __name__ == "__main__":
    asyncio.run(main())