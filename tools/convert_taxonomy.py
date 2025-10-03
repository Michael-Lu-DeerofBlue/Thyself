import argparse
import json
import sys
from pathlib import Path
import yaml
import re

def load_yaml_with_inline_fix(path: Path):
    text = path.read_text(encoding='utf-8')
    original_text = text
    # Strip any C-style block comments just in case
    text = re.sub(r"/\*.*?\*/", "\n", text, flags=re.DOTALL)
    try:
        data = yaml.safe_load(text)
    except Exception:
        # Salvage: quote desc lines that contain additional ':' (likely causing parse issues)
        salvaged_lines = []
        for line in text.splitlines():
            m = re.match(r'^(\s*desc:\s*)(.+)$', line)
            if m:
                prefix, val = m.groups()
                stripped = val.strip()
                if not (stripped.startswith('"') and stripped.endswith('"')) and ': ' in stripped:
                    escaped = stripped.replace('"', '\"')
                    line = f"{prefix}\"{escaped}\""
            salvaged_lines.append(line)
        text = '\n'.join(salvaged_lines)
        data = yaml.safe_load(text)
    # If parse succeeded, attempt to recover full inline desc values containing commas from original text
    if isinstance(data, dict) and 't0' in data:
        parents = data['t0'] or []
    elif isinstance(data, list):
        parents = data
    else:
        return data
    # Build index for quick child lookup
    child_index = {}
    for p in parents:
        for c in p.get('t1', []) or []:
            child_index[(p.get('id'), c.get('id'))] = c
    # Regex to capture inline child entries: - {id: xxx, en: ..., desc: ...}
    inline_re = re.compile(r'-\s*\{([^}]+)\}')
    for line in original_text.splitlines():
        m = inline_re.search(line)
        if not m:
            continue
        inside = m.group(1)
        # Require id: and desc:
        if 'id:' not in inside or 'desc:' not in inside:
            continue
        # Extract id (first occurrence)
        id_m = re.search(r'id:\s*([^,]+)', inside)
        if not id_m:
            continue
        cid = id_m.group(1).strip()
        # Extract desc: everything after 'desc:' up to end (we assume desc is last key in inline map in this file)
        desc_pos = inside.find('desc:')
        if desc_pos == -1:
            continue
        desc_val = inside[desc_pos + len('desc:'):].strip()
        # Remove trailing commas if any
        if desc_val.endswith(','):
            desc_val = desc_val[:-1].rstrip()
        # Remove surrounding quotes if already quoted
        if (desc_val.startswith('"') and desc_val.endswith('"')) or (desc_val.startswith("'") and desc_val.endswith("'")):
            desc_clean = desc_val[1:-1]
        else:
            desc_clean = desc_val
        # Attempt to locate which parent this belongs to by scanning previously parsed children
        for (pid, chid), child_obj in child_index.items():
            if chid == cid:
                # If parsed desc shorter than recovered (likely truncated), replace
                existing = child_obj.get('desc', '') or ''
                if len(desc_clean) > len(existing):
                    child_obj['desc'] = desc_clean
                break
    return data

def normalize(data):
    """Return canonical dict with key 't0' -> list[ {..., t1:[...] } ]."""
    if isinstance(data, dict) and 't0' in data:
        t0 = data['t0'] or []
    elif isinstance(data, list):
        t0 = data
    else:
        raise SystemExit("Input YAML not recognized: expected mapping with key 't0' or a top-level list")
    # Ensure each parent has list t1 (maybe empty)
    for parent in t0:
        if 't1' not in parent or parent['t1'] is None:
            parent['t1'] = []
    return {'t0': t0}

def sort_taxonomy(obj: dict):
    obj['t0'] = sorted(obj['t0'], key=lambda x: x.get('id',''))
    for p in obj['t0']:
        if isinstance(p.get('t1'), list):
            p['t1'] = sorted(p['t1'], key=lambda x: x.get('id',''))
    return obj

def main():
    ap = argparse.ArgumentParser(description='Convert taxonomy YAML (hierarchical) to JSON.')
    ap.add_argument('--input', '-i', default='ml/taxonomies/t0.yaml', help='Path to YAML taxonomy file')
    ap.add_argument('--output', '-o', default='ml/taxonomies/taxonomy.json', help='Destination JSON path')
    ap.add_argument('--no-sort', action='store_true', help='Disable sorting by id')
    ap.add_argument('--minify', action='store_true', help='Output compact JSON (no pretty indent)')
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")
    data = load_yaml_with_inline_fix(in_path)
    norm = normalize(data)
    if not args.no_sort:
        norm = sort_taxonomy(norm)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.minify:
        out_path.write_text(json.dumps(norm, ensure_ascii=False, separators=(',',':')), encoding='utf-8')
    else:
        out_path.write_text(json.dumps(norm, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out_path)

if __name__ == '__main__':
    main()
