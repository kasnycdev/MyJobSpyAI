import os
import subprocess

def update_dependencies():
    # Create/update requirements-docs.txt
    requirements = [
        "mkdocs>=1.5.0",
        "mkdocs-material>=9.0.0",
        "mkdocs-material-extensions>=1.0.0",
        "mkdocs-mermaid2>=0.5.0",
        "pymdown-extensions>=9.0",
        "mkdocs-gen-files>=0.3.0",
        "mkdocs-section-index>=0.3.0",
        "mkdocs-literate-nav>=0.4.0",
        "mkdocs-git-revision-date-localized-plugin>=0.10.0",
        "mkdocs-git-authors-plugin>=0.1.0",
        "mkdocs-with-pdf>=0.1.0",
        "mkdocs-minify-plugin>=0.4.0",
        "mkdocs-awesome-pages-plugin>=2.7.0",
        "mkdocs-autolinks-plugin>=0.0.1",
        "mkdocs-jupyter>=0.1.0",
        "mkdocs-autolinks-plugin>=0.0.1"
    ]

    with open('requirements-docs.txt', 'w') as f:
        f.write('\n'.join(requirements))

    # Install dependencies
    subprocess.run(['pip', 'install', '-r', 'requirements-docs.txt'])

def main():
    update_dependencies()
    print("Documentation dependencies updated successfully!")

if __name__ == "__main__":
    main()
