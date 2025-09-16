#!/usr/bin/env python
import json, argparse, os, subprocess, tempfile, sys

"""Extract titles from exported extension JSON and run inference to produce results.jsonl.
Expected input: a JSON array of event objects (export from dashboard) with fields 'title' and 'url'.
We filter for YouTube ("youtube.com/watch") and Bilibili ("bilibili.com/video") page view events (those lacking 'type' or with type not video_*).
"""

def unique_order(seq):
    seen = set(); out = []
    for x in seq:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def main(args):
    data = json.load(open(args.input, encoding='utf-8'))
    titles = []
    for evt in data:
        url = evt.get('url','')
        if 'youtube.com/watch' in url or 'bilibili.com/video' in url:
            t = evt.get('title','').strip()
            if t:
                titles.append(t)
    titles = unique_order(titles)
    if not titles:
        print("No titles found.", file=sys.stderr)
        return 1
    # Call infer.py
    cmd = [sys.executable, 'ml/src/infer.py', '--titles', *titles[:args.limit]]
    if args.supervised:
        cmd.append('--supervised')
    res = subprocess.check_output(cmd, text=True)
    ranked = json.loads(res)
    with open(args.output, 'w', encoding='utf-8') as f:
        for t, r in zip(titles[:args.limit], ranked):
            f.write(json.dumps({'title': t, 'ranked': r}, ensure_ascii=False) + '\n')
    print(f"Wrote {len(ranked)} rows to {args.output}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Path to exported local-events.json')
    ap.add_argument('--output', default='results.jsonl')
    ap.add_argument('--limit', type=int, default=100)
    ap.add_argument('--supervised', action='store_true')
    sys.exit(main(ap.parse_args()) or 0)
