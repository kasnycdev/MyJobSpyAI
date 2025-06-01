"""Tests for the ResumeParserV2 with DocLing integration."""
import asyncio
import os
import logging
from pathlib import Path
import pytest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the test data directory
TEST_DATA_DIR = Path(__file__).parent / "data" / "resumes"

# Skip tests if test data directory doesn't exist
if not TEST_DATA_DIR.exists():
    pytest.skip("Test data directory not found", allow_module_level=True)

# Get all test files
test_files = list(TEST_DATA_DIR.glob("*.*"))

# Filter for supported file extensions
SUPPORTED_EXTENSIONS = {
    '.pdf', '.docx', '.doc', '.rtf', '.odt', '.txt', '.html', '.htm'
}
test_files = [f for f in test_files if f.suffix.lower() in SUPPORTED_EXTENSIONS]

# Skip if no test files found
if not test_files:
    pytest.skip("No test files found in test data directory", allow_module_level=True)

# Import the parser
from myjobspyai.parsers import ResumeParser, parse_resume


@pytest.mark.asyncio
async def test_parse_resume():
    """Test parsing a resume file using the new parser."""
    test_file = test_files[0]  # Test with the first available file
    logger.info(f"Testing with file: {test_file}")
    
    # Test with the class-based approach
    parser = ResumeParser()
    text = await parser.parse_resume(test_file)
    
    # Basic validation
    assert isinstance(text, str)
    assert len(text) > 0
    assert len(text.split()) > 10  # At least 10 words
    
    # Test with the function-based approach for backward compatibility
    text_func = await parse_resume(test_file)
    assert isinstance(text_func, str)
    assert len(text_func) > 0
    
    logger.info(f"Successfully extracted {len(text)} characters from {test_file}")


@pytest.mark.asyncio
async def test_parse_all_supported_formats():
    """Test parsing all supported file formats."""
    parser = ResumeParser()
    
    # Group files by extension
    files_by_ext = {}
    for ext in SUPPORTED_EXTENSIONS:
        ext_files = [f for f in test_files if f.suffix.lower() == ext]
        if ext_files:
            files_by_ext[ext] = ext_files[0]  # Just test one file per extension
    
    # Test each supported format
    for ext, test_file in files_by_ext.items():
        logger.info(f"Testing {ext.upper()} file: {test_file}")
        try:
            text = await parser.parse_resume(test_file)
            
            # Basic validation
            assert isinstance(text, str), f"Expected string, got {type(text)} for {test_file}"
            assert len(text) > 0, f"Empty text extracted from {test_file}"
            assert len(text.split()) > 5, f"Not enough text extracted from {test_file}"
            
            logger.info(f"✓ Successfully parsed {ext.upper()} file: {len(text)} characters")
            
        except Exception as e:
            logger.error(f"Failed to parse {test_file}: {str(e)}", exc_info=True)
            pytest.fail(f"Failed to parse {test_file}: {str(e)}")


@pytest.mark.asyncio
async def test_invalid_file():
    """Test parsing a non-existent file."""
    parser = ResumeParser()
    
    # Test with non-existent file
    non_existent = TEST_DATA_DIR / "nonexistent_file.pdf"
    text = await parser.parse_resume(non_existent)
    assert text == ""
    
    # Test with unsupported file type
    unsupported = TEST_DATA_DIR / "unsupported_file.xyz"
    if unsupported.exists():
        text = await parser.parse_resume(unsupported)
        assert text == ""


if __name__ == "__main__":
    # Run the tests
    import sys
    sys.exit(pytest.main(["-v", "-s", __file__]))
