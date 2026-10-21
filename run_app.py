import os
from pathlib import Path
import sys
import traceback

if __name__ == "__main__":
    try:
        from app.main import main
        main()
    except Exception:
        folder = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Personal Denoiser"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "startup-error.log").write_text(traceback.format_exc(), encoding="utf-8")
        if "--smoke-test" in sys.argv:
            sys.exit(1)
        raise
