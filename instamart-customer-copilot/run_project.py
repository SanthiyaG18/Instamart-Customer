"""Run all project stages in sequence after installing requirements."""
from pathlib import Path
import subprocess
import sys
from src import (
    part_a_clean,
    part_a_sql,
    part_a_validate,
    part_b_lapse,
    part_b_segments,
    part_c_triage,
    part_d_agent,
)

ROOT = Path(__file__).resolve().parent


def main():
    print("\n=== PREPARE: GENERATE DATA ===", flush=True)
    subprocess.run([sys.executable, str(ROOT / "generate_data.py")], cwd=ROOT, check=True)

    print("\n=== PART A: AUDIT / CLEAN / VALIDATE / SQL ===", flush=True)
    part_a_clean.main()
    part_a_validate.main()
    part_a_sql.main()

    print("\n=== PART B: SEGMENTS / LAPSE MODEL ===", flush=True)
    part_b_segments.main()
    part_b_lapse.main()

    print("\n=== PART C: TICKET TRIAGE ===", flush=True)
    part_c_triage.main()

    print("\n=== PART D: POLICY RAG / REFUND AGENT ===", flush=True)
    part_d_agent.main()

    print("\nProject pipeline finished. Review outputs/ and data/instamart.db.")


if __name__ == "__main__":
    main()
