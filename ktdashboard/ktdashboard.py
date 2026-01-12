#!/usr/bin/env python
import argparse, subprocess, sys, os


def main():
    parser = argparse.ArgumentParser(prog="ktdashboard")
    parser.add_argument(
        "--backend",
        choices=["panel", "streamlit"],
        default="panel",
        help="Backend to use for visualization",
    )
    parser.add_argument(
        "filename", nargs="?", help="Path to cache JSON file (optional for streamlit)"
    )

    args = parser.parse_args()

    if args.backend == "panel":
        if not args.filename:
            print("Cachefile is required for the 'panel' backend")
            exit(1)
        if not os.path.isfile(args.filename):
            print("Cachefile not found")
            exit(1)

    if (
        args.backend == "streamlit"
        and args.filename
        and not os.path.isfile(args.filename)
    ):
        print("Cachefile not found")
        exit(1)

    if args.backend == "streamlit":
        script_path = os.path.join(os.path.dirname(__file__), "streamlit_dashboard.py")
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            script_path,
            "--",
        ]
        # pass filename only when provided
        if args.filename:
            cmd.append(args.filename)
        subprocess.run(cmd)
        return

    if args.backend == "panel":
        from panel_dashboard import serve_panel

        serve_panel(args.filename)
        return

    exit(1)


if __name__ == "__main__":
    main()
