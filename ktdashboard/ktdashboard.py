#!/usr/bin/env python
import argparse, subprocess, sys, os

def main():
    """Command-line interface with backend selection."""

    parser = argparse.ArgumentParser(prog="ktdashboard")
    parser.add_argument("--backend", choices=["panel", "streamlit"], default="panel", help="Backend to use for visualization")
    parser.add_argument("filename", help="Path to cache JSON file")

    args = parser.parse_args()

    if not os.path.isfile(args.filename):
        print("Cachefile not found")
        exit(1)

    if args.backend == "streamlit":
        script_path = os.path.join(os.path.dirname(__file__), "streamlit_dashboard.py")
        cmd = [sys.executable, "-m", "streamlit", "run", script_path, "--", args.filename]
        subprocess.run(cmd)
        return

    if args.backend == "panel":
        from panel_dashboard import serve_panel
        serve_panel(args.filename)
        return

    exit(1)



if __name__ == "__main__":
    main()
