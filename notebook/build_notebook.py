import json, re, ast, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
parts = [open(os.path.join(HERE, 'src', f'p{i}.py'), encoding='utf-8').read() for i in range(1, 6)]
src = '\n'.join(parts)
val = open(os.path.join(HERE, '..', 'reference', 'validate_submission.py'), encoding='utf-8').read()
assert "'''" not in val
src = src.replace('__OFFICIAL_VALIDATOR__', val)
cells = []
for block in re.split(r'^# %%', src, flags=re.M):
    if not block.strip():
        continue
    if block.startswith(' [markdown]'):
        body = block[len(' [markdown]'):].strip('\n')
        lines = [l[2:] if l.startswith('# ') else (l[1:] if l.startswith('#') else l) for l in body.split('\n')]
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': '\n'.join(lines)})
    else:
        body = block.strip('\n')
        ast.parse(body)          # syntax check per cell
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': body})
for c in cells:
    txt = c['source']
    c['source'] = [l + '\n' for l in txt.split('\n')]
    c['source'][-1] = c['source'][-1].rstrip('\n')
nb = {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10'},
        'kaggle': {'accelerator': 'gpu', 'isInternetEnabled': False, 'isGpuEnabled': True}},
      'nbformat': 4, 'nbformat_minor': 4}
out = os.path.join(HERE, 'business_entity_resolution.ipynb')
json.dump(nb, open(out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
open(os.path.join(HERE, 'all_code_for_lint.py'), 'w', encoding='utf-8').write('\n\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'code'))
print(out, 'cells:', len(cells), 'code:', sum(c['cell_type'] == 'code' for c in cells))
