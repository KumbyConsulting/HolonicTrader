import os
import re
import importlib.util
import sys

# Path to project root
PROJECT_ROOT = os.getcwd()

def get_config_module():
    """Load config module dynamically"""
    try:
        spec = importlib.util.spec_from_file_location("config", os.path.join(PROJECT_ROOT, "config.py"))
        config = importlib.util.module_from_spec(spec)
        sys.modules["config"] = config
        spec.loader.exec_module(config)
        return config
    except Exception as e:
        print(f"❌ Failed to load config.py: {e}")
        sys.exit(1)

def scan_codebase_for_config_usage(root_dir):
    """Scan all .py files for config.VARIABLE usage"""
    config_vars = set()
    pattern = re.compile(r'config\.([A-Z_][A-Z0-9_]*)')
    
    print(f"🔍 Scanning codebase in {root_dir}...")
    
    for root, dirs, files in os.walk(root_dir):
        if 'venv' in root or '__pycache__' in root:
            continue
            
        for file in files:
            if file.endswith('.py') and file != 'config.py':
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        matches = pattern.findall(content)
                        for match in matches:
                            config_vars.add(match)
                except Exception as e:
                    print(f"⚠️ Could not read {file}: {e}")
                    
    return config_vars

def verify_config_integrity():
    config = get_config_module()
    used_vars = scan_codebase_for_config_usage(PROJECT_ROOT)
    
    missing = []
    present = []
    
    print(f"\n📋 Found {len(used_vars)} unique config references in codebase.")
    print("-" * 50)
    
    for var in sorted(used_vars):
        if hasattr(config, var):
            present.append(var)
        else:
            missing.append(var)
            print(f"❌ MISSING in config.py: {var}")
            
    if missing:
        print("-" * 50)
        print(f"🚨 CRITICAL FAILURE: {len(missing)} config attributes are missing!")
        print("These will cause AttributeError if hit during execution.")
        sys.exit(1)
    else:
        print(f"✅ ALL CLEAR. All {len(present)} referenced keys exist in config.py.")
        sys.exit(0)

if __name__ == "__main__":
    verify_config_integrity()
