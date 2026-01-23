#!/usr/bin/env python3
"""Force fix all manifest and config files with simple string replacement."""

from pathlib import Path

# Simple string replacements (order matters - do longer patterns first)
REPLACEMENTS = [
    ("g_cities_loc", "g_city_locations"),
    ("g_defs", "g_word_definitions"),
    ("g_drugs", "g_med_indications"),
    ("cities_loc", "city_locations"),
    ("defs", "word_definitions"),
    ("drugs", "med_indications"),
]


def fix_file(file_path: Path, dry_run: bool = True) -> bool:
    """Fix a file using simple string replacement."""
    try:
        with open(file_path) as f:
            content = f.read()
        
        original = content
        
        # Apply all replacements
        for old, new in REPLACEMENTS:
            content = content.replace(old, new)
        
        if content != original:
            if dry_run:
                print(f"Would update: {file_path.relative_to(Path.cwd())}")
            else:
                with open(file_path, 'w') as f:
                    f.write(content)
                print(f"✓ Updated: {file_path.relative_to(Path.cwd())}")
            return True
        
        return False
    except Exception as e:
        print(f"✗ Error: {file_path}: {e}")
        return False


def main(dry_run: bool = True):
    """Fix all files."""
    base_path = Path.cwd() / "outputs" / "probes"
    
    if not base_path.exists():
        print(f"Error: {base_path} not found")
        return
    
    print(f"{'=' * 60}")
    print("Force Fix All Manifests and Configs")
    print(f"{'=' * 60}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE RUN'}")
    print(f"{'=' * 60}\n")
    
    # Find all files
    json_files = list(base_path.rglob("*.json"))
    yaml_files = list(base_path.rglob("*.yaml"))
    
    all_files = json_files + yaml_files
    print(f"Found {len(all_files)} files to check\n")
    
    updated = 0
    for file_path in all_files:
        if fix_file(file_path, dry_run):
            updated += 1
    
    print(f"\n{'=' * 60}")
    if dry_run:
        print(f"Would update {updated} files")
        print("\nRun with --apply:")
        print("  python scripts/force_fix_manifests.py --apply")
    else:
        print(f"✓ Updated {updated} files")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    main(dry_run=not args.apply)
