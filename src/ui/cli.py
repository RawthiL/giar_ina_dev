import subprocess
import sys
from pathlib import Path


def run_streamlit():
    """
    Launch the Streamlit app via Poetry script.
    """
    app_path = Path(__file__).resolve().parent / "app.py"
    print(app_path)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
        ],
        check=True,
    )
