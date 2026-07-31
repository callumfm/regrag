"""Export the OpenAPI schema for frontend type generation."""

import json
from pathlib import Path

from app.main import create_app

out = Path(__file__).parents[2] / "frontend" / "openapi.json"
out.write_text(json.dumps(create_app().openapi(), indent=2) + "\n")
print(f"wrote {out}")
