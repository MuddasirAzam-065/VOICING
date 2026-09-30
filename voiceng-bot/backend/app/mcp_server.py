import os

from mcp.server.fastmcp import FastMCP

from .kb import db, search


PORT = int(
    os.getenv(
        "PORT",
        "9000",
    )
)


mcp = FastMCP(
    "voiceng-kb",
    host="0.0.0.0",
    port=PORT,
)


@mcp.tool()
def search_knowledge(
    query: str,
    persona: str = "voiceng",
    top_k: int = 5,
) -> list[dict]:

    organization = (
        "FIA"
        if persona == "fia"
        else "VoiceNG"
    )

    hits = search(
        f"{organization} {query}",
        persona,
        top_k,
    )

    return [
        {
            "title": hit.get(
                "title",
                "",
            ),
            "section": hit.get(
                "section",
                "",
            ),
            "url": hit.get(
                "url",
                "",
            ),
            "score": round(
                float(
                    hit.get(
                        "score",
                        0,
                    )
                ),
                3,
            ),
            "text": hit.get(
                "text",
                "",
            ),
        }
        for hit in hits
    ]


@mcp.tool()
def list_documents(
    persona: str = "voiceng",
) -> list[dict]:

    return list(
        db.documents.find(
            {
                "persona": persona,
            },
            {
                "_id": 0,
                "slug": 1,
                "title": 1,
                "url": 1,
            },
        )
    )


@mcp.tool()
def get_document(
    persona: str,
    slug: str,
) -> str:

    document = db.documents.find_one(
        {
            "persona": persona,
            "slug": slug,
        },
        {
            "_id": 0,
            "content": 1,
        },
    )

    if not document:
        return "Document not found."

    return document.get(
        "content",
        "",
    )


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http"
    )