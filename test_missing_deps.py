#!/usr/bin/env python3
"""
Test dependency checking with simulated missing dependencies
"""
import sys
from pathlib import Path

# Simulate missing dependencies by patching the modules
original_modules = dict(sys.modules)
sys.modules['geopandas'] = None
sys.modules['fiona'] = None

# Remove any cached imports
modules_to_remove = []
for module in sys.modules:
    if module and ('geopandas' in module or 'fiona' in module):
        modules_to_remove.append(module)

for module in modules_to_remove:
    if module in sys.modules:
        del sys.modules[module]

# Now test the dependency checker
test_dir = Path(__file__).parent / "tests"
if str(test_dir) not in sys.path:
    sys.path.insert(0, str(test_dir))

from conftest import DependencyChecker

print("Testing with simulated missing dependencies (geopandas, fiona):")
print("=" * 60)

checker = DependencyChecker()
report = checker.generate_report()
print(report)

# Restore original modules
sys.modules.update(original_modules)
