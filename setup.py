from setuptools import setup, find_packages

# Read requirements from pyproject.toml
with open('pyproject.toml', 'r', encoding='utf-8') as f:
    content = f.read()

# Extract dependencies from pyproject.toml
install_requires = []
for line in content.split('\n'):
    if 'dependencies' in line:
        continue
    if line.strip().startswith('"') and '"' in line[1:]:
        dep = line.strip().strip('"\',').strip()
        if dep and not dep.startswith('['):
            install_requires.append(dep)

setup(
    name="myjobspyai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'myjobspyai=myjobspyai.__main__:main',
        ],
    },
    python_requires='>=3.8',
)
