"""Read-only verification of the saved final DOCX and retained research sources."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from docx import Document
import hashlib
import json

HERE = Path(__file__).resolve().parent
ARTICLE = HERE.parents[2]
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
repairs = json.loads((HERE / 'document_repairs.json').read_text())
target = Path(repairs['target'])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(target) == repairs['after_sha256']
assert sha(repairs['backup']) == repairs['before_sha256']


def doc(path):
    with ZipFile(path) as z:
        assert z.testzip() is None
        parts = {n: z.read(n) for n in z.namelist()}
    return parts, E.fromstring(parts['word/document.xml']).find('w:body', NS)


before, old = doc(repairs['backup'])
after, new = doc(target)
assert before.keys() == after.keys()
assert all(before[k] == after[k] for k in before if k != 'word/document.xml')
assert len(old) == len(new)
text = lambda node: ''.join(t.text or '' for t in node.findall('.//w:t', NS))
canonical = lambda node: E.tostring(node, method='c14n', exclusive=True)
changed = set()
for edit in repairs['edits']:
    if edit['location'].startswith('P'):
        i = int(edit['location'][1:])
        assert text(old[i]) == edit['before']
        assert text(new[i]) == edit['after']
        changed.add(i)
    else:
        _, row, col = edit['location'].split('/')
        a = old[87].findall('w:tr', NS)[int(row[1:])].findall('w:tc', NS)[int(col[1:])]
        b = new[87].findall('w:tr', NS)[int(row[1:])].findall('w:tc', NS)[int(col[1:])]
        assert text(a) == edit['before'] and text(b) == edit['after']
        changed.add(87)
for i in range(len(old)):
    if i not in changed:
        assert canonical(old[i]) == canonical(new[i])
old_tables, new_tables = old.findall('w:tbl', NS), new.findall('w:tbl', NS)
assert len(old_tables) == len(new_tables) == len(Document(target).tables) == 13
assert all(canonical(a) == canonical(b) for a, b in zip(old_tables[1:], new_tables[1:]))
for i in range(268, 292):
    assert canonical(old[i]) == canonical(new[i])
sources_checked = 0
for p, expected in repairs['source_sha256'].items():
    assert sha(p) == expected, p
    sources_checked += 1
previous = json.loads((ARTICLE / 'results/document_edits/article_update_20261027_v1/manifest.json').read_text())
for source in previous['sources']:
    p = ARTICLE.parent / source['path']
    assert sha(p) == source['sha256'], p
    sources_checked += 1
manifest_files_checked = 0
for folder in ['runs/hierarchy_full_20261026_v1', 'runs/context_full_20261027_v2',
               'analyses/context_v2_interpretation_20261027_v1',
               'analyses/context_sufficiency_partial_20261027_v1']:
    root = ARTICLE / 'results' / folder
    m = json.loads((root / 'manifest.json').read_text())
    for relative, expected in m.get('artifact_sha256', {}).items():
        assert sha(root / relative) == expected, root / relative
        manifest_files_checked += 1
result = {'status': 'passed', 'final_docx_sha256': sha(target),
          'word_package_valid': True, 'opens_with_python_docx': True,
          'table_count': 13, 'tables_2_through_13_unchanged': True,
          'bibliography_24_entries_unchanged': True,
          'all_27_corrections_present': len(repairs['edits']) == 27,
          'all_unselected_elements_and_other_package_parts_unchanged': True,
          'source_hash_checks': sources_checked,
          'run_manifest_artifact_hash_checks': manifest_files_checked,
          'visual_rendering': 'not performed: Word/LibreOffice unavailable',
          'limits': 'Replay of saved outputs is not a new retrieval/encoder/CPU experiment or human validation'}
(HERE / 'final_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False))
