import json
import os
import sys
import argparse
import glob
from collections import Counter

# --- CONFIGURATION ---
DEFAULT_DIR = "/mnt/c/rag_data"

def get_file_selection():
    parser = argparse.ArgumentParser(description="Inspect JSON structure and extract topics.")
    parser.add_argument("file_path", nargs="?", help="Path to the JSON file")
    args = parser.parse_args()

    if args.file_path:
        return args.file_path

    print(f"Scanning {DEFAULT_DIR} for JSON files...")
    files = glob.glob(os.path.join(DEFAULT_DIR, "*.json"))
    
    if not files:
        print(f"No JSON files found in {DEFAULT_DIR}")
        sys.exit(1)

    print("\nAvailable Files:")
    for i, f in enumerate(files):
        print(f"[{i+1}] {os.path.basename(f)}")

    while True:
        try:
            sel = input("\nSelect a file number: ").strip()
            if sel.lower() == 'q': sys.exit(0)
            idx = int(sel) - 1
            if 0 <= idx < len(files): return files[idx]
        except ValueError: pass

def extract_topics(file_path):
    print(f"\n--- ANALYZING TOPICS IN: {os.path.basename(file_path)} ---")
    
    # We will count frequencies of Subjects and Objects
    subjects = Counter()
    objects = Counter()
    relations = Counter()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Recursive hunter to find all triples deeply nested
        stack = [data]
        count = 0
        
        while stack:
            current = stack.pop()
            
            if isinstance(current, dict):
                # Check for Semantic Keys
                s = current.get('subject') or current.get('head') or current.get('arg1')
                r = current.get('relation') or current.get('type') or current.get('predicate')
                o = current.get('object') or current.get('tail') or current.get('arg2')
                
                if s and o:
                    subjects[str(s).lower().strip()] += 1
                    objects[str(o).lower().strip()] += 1
                    if r: relations[str(r).lower().strip()] += 1
                    count += 1
                
                # Dig deeper
                for v in current.values():
                    if isinstance(v, (dict, list)): stack.append(v)
                    
            elif isinstance(current, list):
                # Check for List Triples ['Sub', 'Rel', 'Obj']
                if len(current) == 3 and all(isinstance(x, str) for x in current):
                    subjects[str(current[0]).lower().strip()] += 1
                    relations[str(current[1]).lower().strip()] += 1
                    objects[str(current[2]).lower().strip()] += 1
                    count += 1
                else:
                    for item in current:
                        if isinstance(item, (dict, list)): stack.append(item)

        print(f"✅ Scanned {count} facts found in the file.")
        
        if count > 0:
            print("\n=== TOP 10 TOPICS (Most Frequent Subjects) ===")
            for sub, freq in subjects.most_common(10):
                print(f"{freq:4d} | {sub}")

            print("\n=== TOP 10 RELATIONS (Types of Connections) ===")
            for rel, freq in relations.most_common(10):
                print(f"{freq:4d} | {rel}")
                
            print("\n=== TOP 10 OBJECTS (Common Targets) ===")
            for obj, freq in objects.most_common(10):
                print(f"{freq:4d} | {obj}")
        else:
            print("⚠️  No topics found. The file structure might be empty or unique.")

    except Exception as e:
        print(f"Error analyzing file: {e}")

if __name__ == "__main__":
    target = get_file_selection()
    extract_topics(target)
