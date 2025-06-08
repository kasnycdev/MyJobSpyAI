def get_version():
    """Get the current version from pyproject.toml"""
    import toml
    with open('../pyproject.toml', 'r') as f:
        pyproject = toml.load(f)
    return pyproject['tool']['poetry']['version']

def get_release_date():
    """Get the current date"""
    from datetime import datetime
    return datetime.now().strftime('%B %d, %Y')
