#!/usr/bin/env python3
"""
Extract first names from entities_50k.json
"""
import json

def extract_first_names(input_file, output_file):
    """
    Extract first names from the entities JSON file

    Args:
        input_file: Path to the input JSON file
        output_file: Path to save the first names (one per line)
    """
    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        entities = json.load(f)

    # Extract first names
    first_names = []
    for entity in entities:
        if 'name' in entity and entity['name']:
            # Split by space and take the first part
            first_name = entity['name'].split()[0]
            first_names.append(first_name)

    # Save to output file (one name per line)
    with open(output_file, 'w', encoding='utf-8') as f:
        for name in first_names:
            f.write(name + '\n')

    print(f"Extracted {len(first_names)} first names from {input_file}")
    print(f"Saved to {output_file}")

    # Print first 10 as a sample
    print("\nFirst 10 names:")
    for name in first_names[:10]:
        print(f"  - {name}")

if __name__ == "__main__":
    extract_first_names('../data/entities_50k.json', '../data/first_names_50k.txt')
