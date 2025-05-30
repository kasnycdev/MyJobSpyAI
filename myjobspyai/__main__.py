"""
Main entry point for the myjobspyai package.

This module allows the package to be run directly using `python -m myjobspyai`.
"""

def main():
    """Run the main function from main.py."""
    import sys
    import os
    
    # Add the project root to the Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from main import main as main_func
    import asyncio
    
    try:
        return asyncio.run(main_func())
    except KeyboardInterrupt:
        print("\nExecution cancelled by user.")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
