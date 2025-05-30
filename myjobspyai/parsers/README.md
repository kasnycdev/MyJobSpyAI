# Document Parsers

This module provides document parsing functionality for the MyJobSpyAI application, with a focus on resume/CV processing.

## Resume Parser (v2)

The `ResumeParserV2` class provides an improved implementation of the resume parser that leverages the DocLing library for better document processing.

### Features

- **Multiple Format Support**: Handles PDF, DOCX, DOC, RTF, ODT, TXT, HTML
- **Advanced Processing**: Uses DocLing for better text extraction and formatting
- **Async/Await**: Built with async/await for better performance with I/O operations
- **Error Handling**: Comprehensive error handling and validation
- **Observability**: Integrated with OpenTelemetry for tracing and monitoring

### Usage

```python
from myjobspyai.parsers import ResumeParser, parse_resume
import asyncio

# Using the class-based approach (recommended)
async def parse_resume_file(file_path):
    parser = ResumeParser()
    text = await parser.parse_resume(file_path)
    print(f"Extracted text: {text[:200]}...")  # Print first 200 chars

# Using the function-based approach (legacy compatibility)
async def parse_resume_legacy(file_path):
    text = await parse_resume(file_path)
    print(f"Extracted text: {text[:200]}...")

# Example usage
if __name__ == "__main__":
    file_path = "path/to/your/resume.pdf"
    asyncio.run(parse_resume_file(file_path))
```

### Configuration

The `ResumeParser` constructor accepts an optional configuration dictionary:

```python
config = {
    "timeout": 30,  # seconds
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "allowed_extensions": [".pdf", ".docx", ".doc", ".rtf", ".odt", ".txt", ".html", ".htm"],
}
parser = ResumeParser(config=config)
```

### Methods

#### `parse_resume(file_path, output_format='txt', **kwargs)`

Parse a resume file and extract text.

**Parameters:**
- `file_path` (str/Path): Path to the resume file
- `output_format` (str): Desired output format (default: 'txt')
- `**kwargs`: Additional arguments to pass to the document converter

**Returns:**
- str: Extracted text content, or empty string on error

### Error Handling

The parser includes comprehensive error handling and will return an empty string if the file cannot be parsed. Check the logs for detailed error information.

## Legacy Parser

The legacy parser is still available for backward compatibility:

```python
from myjobspyai.parsers import parse_resume_legacy

async def example():
    text = await parse_resume_legacy("resume.pdf")
    print(text)
```

## Testing

Run the tests with pytest:

```bash
pytest tests/test_resume_parser_v2.py -v
```

## Dependencies

- docling
- python-docx
- pypdf
- textract
- beautifulsoup4
- opentelemetry-api
