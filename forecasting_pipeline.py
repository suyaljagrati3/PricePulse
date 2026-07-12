"""Command-line entry point for the modular commodity forecasting workflow."""

from pathlib import Path
from runpy import run_path


def main():
    """Run the complete legacy-compatible workflow from its package module."""
    workflow = Path(__file__).with_name("forecasting_modules") / "legacy_batch_workflow.py"
    run_path(str(workflow), run_name="__main__")


if __name__ == "__main__":
    main()
