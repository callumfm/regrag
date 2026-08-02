from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    corpus_version: str | None
