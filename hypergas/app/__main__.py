import os
import subprocess
import sys

def main():
    """Launch the Streamlit application."""
    app_dir = os.path.dirname(__file__)
    about_py = os.path.join(app_dir, "About.py")

    # launch Streamlit using same Python environment
    cmd = [sys.executable, "-m", "streamlit", "run", about_py]
    subprocess.run(cmd)
