"""Print sample D-V2 docs to inspect text format for query expansion design."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_courses
from src.doc_builders import build_d_v2

courses = load_courses(str(ROOT / "data" / "1142.db"))
docs = build_d_v2(courses)

# 找幾個有 weekday/lang/unit 的範例
samples = []
for d in docs:
    if "上課時間: 星期" in d.text and "語言:" in d.text:
        samples.append(d)
        if len(samples) >= 3: break

for d in samples:
    print(f"--- {d.course_id} ---")
    print(d.text[:600])
    print()
