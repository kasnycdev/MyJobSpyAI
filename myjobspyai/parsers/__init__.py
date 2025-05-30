"""Document parsers for the MyJobSpyAI application.

This package provides various parsers for extracting text from different document formats.
"""

# Legacy imports for backward compatibility
from .resume_parser import parse_resume as legacy_parse_resume

# New parser with DocLing integration
from .resume_parser_v2 import ResumeParser, parse_resume

# For backward compatibility, expose the legacy parse_resume function
parse_resume_legacy = legacy_parse_resume

__all__ = [
    'ResumeParser',
    'parse_resume',
    'parse_resume_legacy',
]
