import pytest
import os
from backend.scripts.labeling import load_label_debug_module

def test_label_debug_file_exists():
    # Ensure the file exists before testing
    debug_file = 'backend/scripts/label_debug.py'
    assert os.path.exists(debug_file), f"File {debug_file} does not exist"
    
    # Test loading the module
    module = load_label_debug_module()
    assert module is not None

def test_label_debug_file_missing():
    # Temporarily rename file to simulate missing
    debug_file = 'backend/scripts/label_debug.py'
    backup_file = debug_file + '.backup'
    
    if os.path.exists(debug_file):
        os.rename(debug_file, backup_file)
    
    try:
        with pytest.raises(FileNotFoundError):
            load_label_debug_module()
    finally:
        # Restore file
        if os.path.exists(backup_file):
            os.rename(backup_file, debug_file)