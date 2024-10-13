import sys
from pathlib import Path
import vcr

def scrub_api_key(request):
    if 'Authorization' in request.headers:
        request.headers['Authorization'] = 'REDACTED'
    return request

my_vcr = vcr.VCR(
    filter_headers=['Authorization'],
    before_record_request=scrub_api_key,
)

# Add the project root directory to Python's path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))