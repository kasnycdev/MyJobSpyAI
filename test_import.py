import sys
print("Python path:")
for path in sys.path:
    print(path)

try:
    import instructor
    print("Successfully imported instructor")
except ImportError as e:
    print(f"Failed to import instructor: {e}")
