import sys
from pathlib import Path

# In case Render Root Directory is set to 'web_ui', resolve project root
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demo.server import app, run_server

if __name__ == "__main__":
    run_server()
