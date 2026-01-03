#!/usr/bin/env python3
"""Rename datapack folders and update all internal path references.

This script:
1. Finds folders with old datapack names (defs, cities_loc, etc.)
2. Updates paths in config.yaml files within those folders
3. Updates paths in all manifest JSON files
4. Renames the folders to new names
"""

import json
import re
import shutil
from pathlib import Path

# Define the renaming mappings (only include folders that need renaming)
FOLDER_MAPPINGS = {
    "defs": "word_definitions",
    "cities_loc": "city_locations",
    "drugs": "med_indications",
    # Add g_ prefixed versions that need renaming
    "g_defs": "g_word_definitions",
    "g_cities_loc": "g_city_locations",
    "g_drugs": "g_med_indications",
    # Note: med_indications and g_med_indications are already correct, no need to rename
}

# Path patterns to update in files (includes both with and without g_ prefix)
PATH_PATTERNS = {
    "defs": "word_definitions",
    "cities_loc": "city_locations",
    "drug": "med_indications",
    "g_defs": "g_word_definitions",
    "g_cities_loc": "g_city_locations",
    "g_drug": "g_med_indications",
}


def find_folders_to_rename(base_path: Path, dry_run: bool = True) -> list:
    """Find all folders that need to be renamed."""
    folders_to_rename = []
    
    # Find all directories recursively
    for dir_path in base_path.rglob("*"):
        if not dir_path.is_dir():
            continue
        
        # Skip folders named "prompt" or inside a "prompt" folder
        if "prompt" in dir_path.parts:
            continue
        
        folder_name = dir_path.name
        
        # Check if folder name contains any old names that need replacement
        for old_name, new_name in FOLDER_MAPPINGS.items():
            # Check if the old name appears in the folder name
            if old_name in folder_name:
                # Replace old name with new name in the folder name
                new_folder_name = folder_name.replace(old_name, new_name)
                new_path = dir_path.parent / new_folder_name
                
                # Only add if it actually changes the name and target doesn't exist
                if new_folder_name != folder_name:
                    folders_to_rename.append((dir_path, new_path))
                    if dry_run:
                        print(f"Would rename: {dir_path.relative_to(base_path)}")
                        print(f"         to: {new_path.relative_to(base_path)}")
                    break  # Only match once per folder
    
    return folders_to_rename


def update_yaml_file(file_path: Path, dry_run: bool = True) -> bool:
    """Update paths in a YAML config file."""
    try:
        with open(file_path) as f:
            content = f.read()
        
        original_content = content
        
        # Replace old paths with new ones
        for old_name, new_name in PATH_PATTERNS.items():
            content = re.sub(
                rf'\b{old_name}\b',
                new_name,
                content
            )
        
        if content != original_content:
            if dry_run:
                print(f"  Would update YAML: {file_path.name}")
            else:
                with open(file_path, 'w') as f:
                    f.write(content)
                print(f"  Updated YAML: {file_path.name}")
            return True
        
        return False
    except Exception as e:
        print(f"  Error updating {file_path}: {e}")
        return False


def update_json_file(file_path: Path, dry_run: bool = True) -> bool:
    """Update paths in a JSON manifest file."""
    try:
        with open(file_path) as f:
            data = json.load(f)
        
        original_str = json.dumps(data, sort_keys=True)
        
        # Recursively update all string values in the JSON
        def update_strings(obj):
            if isinstance(obj, dict):
                return {k: update_strings(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [update_strings(item) for item in obj]
            elif isinstance(obj, str):
                result = obj
                for old_name, new_name in PATH_PATTERNS.items():
                    result = re.sub(rf'\b{old_name}\b', new_name, result)
                return result
            else:
                return obj
        
        updated_data = update_strings(data)
        updated_str = json.dumps(updated_data, sort_keys=True)
        
        if updated_str != original_str:
            if dry_run:
                print(f"  Would update JSON: {file_path.name}")
            else:
                with open(file_path, 'w') as f:
                    json.dump(updated_data, f, indent=2)
                print(f"  Updated JSON: {file_path.name}")
            return True
        
        return False
    except Exception as e:
        print(f"  Error updating {file_path}: {e}")
        return False


def process_folder(folder_path: Path, dry_run: bool = True):
    """Process all config and manifest files in a folder."""
    print(f"\nProcessing folder: {folder_path.name}")
    
    # Update config.yaml files
    config_files = list(folder_path.rglob("config.yaml")) + list(folder_path.rglob("config.yml"))
    for config_file in config_files:
        update_yaml_file(config_file, dry_run)
    
    # Update manifest JSON files
    manifest_files = list(folder_path.rglob("manifests/*.json")) + list(folder_path.rglob("manifest*.json"))
    for manifest_file in manifest_files:
        update_json_file(manifest_file, dry_run)


def rename_folders(folders_to_rename: list, dry_run: bool = True):
    """Rename all folders from old names to new names."""
    print(f"\n{'=' * 60}")
    print("Renaming folders...")
    print(f"{'=' * 60}")
    
    # Sort by depth (deepest first) to avoid path invalidation
    # Count the number of path separators to determine depth
    folders_to_rename_sorted = sorted(
        folders_to_rename, 
        key=lambda x: len(x[0].parts), 
        reverse=True
    )
    
    for old_path, new_path in folders_to_rename_sorted:
        # Check if path still exists (might have been moved by parent rename)
        if not old_path.exists():
            # Try to find if it was already renamed
            if new_path.exists():
                continue  # Already renamed, skip
            else:
                print(f"⚠️  Skipping {old_path.name}: path no longer exists (likely already renamed)")
                continue
        
        if new_path.exists() and old_path.exists():
            print(f"⚠️  WARNING: {new_path} already exists. Skipping {old_path}")
            continue
        
        if dry_run:
            print(f"Would rename: {old_path}")
            print(f"         to: {new_path}")
        else:
            try:
                shutil.move(str(old_path), str(new_path))
                print(f"✓ Renamed: {old_path.name} -> {new_path.name}")
            except Exception as e:
                print(f"✗ Error renaming {old_path}: {e}")


def main(dry_run: bool = True):
    """Main execution function."""
    base_path = Path(__file__).parent.parent / "outputs" / "probes"
    
    if not base_path.exists():
        print(f"Error: {base_path} does not exist")
        return
    
    print(f"{'=' * 60}")
    print("Datapack Path Renaming Script")
    print(f"{'=' * 60}")
    print(f"Base path: {base_path}")
    print(f"Mode: {'DRY RUN (no changes will be made)' if dry_run else 'LIVE RUN (changes will be applied)'}")
    print(f"{'=' * 60}\n")
    
    # Find all folders to rename
    folders_to_rename = find_folders_to_rename(base_path, dry_run)
    
    if not folders_to_rename:
        print("No folders found to rename.")
        return
    
    print(f"\nFound {len(folders_to_rename)} folders to process.\n")
    
    # Process each folder (update internal files)
    for old_path, new_path in folders_to_rename:
        if old_path.exists():
            process_folder(old_path, dry_run)
    
    # Rename folders
    rename_folders(folders_to_rename, dry_run)
    
    if dry_run:
        print(f"\n{'=' * 60}")
        print("DRY RUN COMPLETE - No changes were made")
        print("Run with --apply flag to apply changes:")
        print("  python scripts/rename_datapack_paths.py --apply")
        print(f"{'=' * 60}")
    else:
        print(f"\n{'=' * 60}")
        print("✓ All changes applied successfully!")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Rename datapack folders and update internal path references"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply the changes (default is dry-run mode)"
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="Override the base path (default: outputs/probes)"
    )
    
    args = parser.parse_args()
    
    dry_run = not args.apply
    main(dry_run=dry_run)
