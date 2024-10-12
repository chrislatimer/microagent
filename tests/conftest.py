import sys
from pathlib import Path

# Add the project root directory to Python's path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))