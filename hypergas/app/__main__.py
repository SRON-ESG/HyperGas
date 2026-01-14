import os
import subprocess
import sys

def main():
    """Launch the Streamlit application."""
    app_dir = os.path.dirname(__file__)
    about_py = os.path.join(app_dir, "About.py")

    # launch Streamlit using same Python environment
    cmd = [sys.executable, "-m", "streamlit", "run", about_py]

    # Add check=True and remove the return to keep process running
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down Streamlit...")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
