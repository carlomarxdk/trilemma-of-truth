#!/usr/bin/env python3
"""Check if manifest files still contain old datapack naming patterns."""

from collections import defaultdict
from pathlib import Path

# Old patterns to check for
OLD_PATTERNS = [
    "defs",
    "cities_loc",
    "drug",
    "g_defs", 
    "g_cities_loc",
    "g_drug",
]

# Patterns that should NOT be flagged (correct names)
CORRECT_PATTERNS = [
    "word_definitions",
    "city_locations", 
    "med_indications",
    "g_word_definitions",
    "g_city_locations",
    "g_med_indications",
]


def check_json_content(file_path: Path) -> list:
    """Check if JSON file contains old patterns.
    
    Returns:
        List of (pattern, location) tuples where old patterns were found.
    """
    try:
        with open(file_path) as f:
            content = f.read()
        
        issues = []
        for pattern in OLD_PATTERNS:
            # Use word boundary check to avoid false positives
            import re

            # Match pattern not followed by underscore (to avoid matching word_definitions when looking for defs)
            matches = re.finditer(rf'\b{pattern}\b(?!_)', content)
            for match in matches:
                # Get context around the match
                start = max(0, match.start() - 20)
                end = min(len(content), match.end() + 20)
                context = content[start:end].replace('\n', ' ')
                issues.append((pattern, context))
        
        return issues
    except Exception as e:
        return [("ERROR", str(e))]


def main():
    """Check all manifest files."""
    base_path = Path.cwd() / "outputs" / "probes"
    
    if not base_path.exists():
        print(f"Error: {base_path} does not exist")
        return
    
    print(f"{'=' * 70}")
    print("Manifest Files Checker")
    print(f"{'=' * 70}\n")
    
    # Find all manifest files
    manifest_files = list(base_path.rglob("manifest*.json"))
    config_files = list(base_path.rglob("config.yaml"))
    
    print(f"Checking {len(manifest_files)} manifest files...")
    print(f"Checking {len(config_files)} config files...\n")
    
    all_issues = defaultdict(list)
    files_with_issues = []
    
    # Check manifests
    for file_path in manifest_files:
        issues = check_json_content(file_path)
        if issues:
            all_issues[str(file_path.relative_to(base_path))] = issues
            files_with_issues.append(file_path)
    
    # Check configs
    for file_path in config_files:
        try:
            with open(file_path) as f:
                content = f.read()
            
            issues = []
            for pattern in OLD_PATTERNS:
                import re
                matches = re.finditer(rf'\b{pattern}\b(?!_)', content)
                for match in matches:
                    start = max(0, match.start() - 20)
                    end = min(len(content), match.end() + 20)
                    context = content[start:end].replace('\n', ' ')
                    issues.append((pattern, context))
            
            if issues:
                all_issues[str(file_path.relative_to(base_path))] = issues
                files_with_issues.append(file_path)
        except:
            pass
    
    # Report results
    if not all_issues:
        print(f"{'=' * 70}")
        print("✓ All files are clean! No old naming patterns found.")
        print(f"{'=' * 70}")
    else:
        print(f"{'=' * 70}")
        print(f"✗ Found {len(all_issues)} files with issues:")
        print(f"{'=' * 70}\n")
        
        for file_path, issues in sorted(all_issues.items()):
            print(f"\n📄 {file_path}")
            for pattern, context in issues:
                print(f"   ❌ Found '{pattern}': ...{context}...")
        
        print(f"\n{'=' * 70}")
        print(f"Summary: {len(all_issues)} files need updates")
        print(f"{'=' * 70}")
        print("\nTo fix these, run:")
        print("  python scripts/fix_remaining_manifests.py --apply")


if __name__ == "__main__":
    main()
