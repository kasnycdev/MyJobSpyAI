"""Main entry point for the MyJobSpy AI package.

This module allows the package to be run as a script using `python -m myjobspyai`.
"""

def main() -> None:
    """Run the main application.

    This function is defined here to avoid circular imports.
    The actual implementation is imported inside the function.
    """
    # Import here to avoid circular imports
    from myjobspyai.main import main as app_main
    app_main()


if __name__ == "__main__":
    main()
