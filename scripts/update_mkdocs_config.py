import yaml
from pathlib import Path

def update_mkdocs_config():
    # Read existing config
    config_path = Path('mkdocs.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update theme configuration
    config['theme'] = {
        'name': 'material',
        'features': [
            'navigation.instant',
            'navigation.tabs',
            'navigation.tabs.sticky',
            'navigation.top',
            'navigation.tracking',
            'search.highlight',
            'search.suggest',
            'search.fuzzy',
            'search.shortcut',
            'toc.follow',
            'toc.integrate',
            'content.code.annotate',
            'content.code.copy',
            'content.code.hilite',
            'content.tabs.link',
            'content.tabs.sticky',
            'content.tooltips',
            'content.code.line-numbers',
            'content.code.wrap',
            'content.code.highlight',
            'content.code.scroll'
        ],
        'palette': {
            'primary': 'blue',
            'accent': 'blue',
            'scheme': 'auto',
            'toggle': {
                'icon': 'material/toggle-switch',
                'name': 'Switch color scheme'
            }
        },
        'font': {
            'text': 'Roboto',
            'code': 'Roboto Mono'
        },
        'logo': 'assets/logo.png',
        'favicon': 'assets/favicon.ico',
        'icon': {
            'repo': 'fontawesome/brands/github',
            'edit': 'material/file-edit-outline',
            'home': 'material/home',
            'menu': 'material/menu',
            'toc': 'material/toc',
            'search': 'material/magnify',
            'prev': 'material/chevron-left',
            'next': 'material/chevron-right',
            'tools': 'material/tools'
        }
    }

    # Update markdown extensions
    config['markdown_extensions'] = [
        'admonition',
        'codehilite',
        'toc',
        'pymdownx.details',
        'pymdownx.emoji',
        'pymdownx.highlight',
        'pymdownx.inlinehilite',
        'pymdownx.magiclink',
        'pymdownx.superfences',
        'pymdownx.tasklist',
        'pymdownx.tabbed',
        'pymdownx.betterem',
        'pymdownx.tilde',
        'pymdownx.caret',
        'pymdownx.keys',
        'pymdownx.mark',
        'pymdownx.progressbar',
        'pymdownx.snippets',
        'pymdownx.smartsymbols',
        'pymdownx.extrarawhtml',
        'pymdownx.tasklist',
        'pymdownx.superfences'
    ]

    # Add plugins
    config['plugins'] = [
        'macros',
        'minify',
        'with-pdf',
        'git-revision-date-localized',
        'git-authors',
        'awesome-pages',
        'autolinks',
        'literate-nav',
        'gen-files',
        'section-index'
    ]

    # Update navigation
    config['nav'] = [
        {
            'Getting Started': [
                'index.md',
                'getting_started/installation.md',
                'getting_started/configuration.md',
                'getting_started/usage.md',
                'getting_started/examples.md'
            ]
        },
        {
            'Features': [
                'features/job_analysis.md',
                'features/resume_matching.md',
                'features/job_scraping.md',
                'features/filtering.md',
                'features/llm_integration.md'
            ]
        },
        {
            'API Reference': [
                'api/index.md',
                'api/myjobspyai.analysis.md',
                'api/myjobspyai.filtering.md',
                'api/myjobspyai.llm.md',
                'api/myjobspyai.scrapers.md',
                'api/myjobspyai.models.md',
                'api/myjobspyai.utils.md'
            ]
        },
        {
            'Development': [
                'development/contributing.md',
                'development/code_style.md',
                'development/testing.md',
                'development/deployment.md'
            ]
        },
        {
            'Enhancement Plans': [
                'enhancement_plans/roadmap.md',
                'enhancement_plans/future_features.md'
            ]
        },
        'Support',
        'License'
    ]

    # Update site information
    config['site_name'] = 'MyJobSpyAI'
    config['site_description'] = 'AI-powered job search and analysis platform'
    config['site_author'] = 'Your Name'
    config['repo_url'] = 'https://github.com/yourusername/myjobspyai'
    config['repo_name'] = 'MyJobSpyAI'
    config['edit_uri'] = 'edit/main/docs/'

    # Update extra features
    config['extra'] = {
        'social': [
            {
                'type': 'github',
                'link': 'https://github.com/yourusername/myjobspyai'
            }
        ],
        'analytics': {
            'provider': 'google',
            'property': 'G-XXXXXX'
        }
    }

    # Write updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print("mkdocs.yml configuration updated successfully!")

if __name__ == "__main__":
    update_mkdocs_config()
