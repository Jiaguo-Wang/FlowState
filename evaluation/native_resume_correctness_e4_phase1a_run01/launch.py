"""复用已经通过CPU门禁的原入口，只将输出根目录改为新的独立归档。"""
from pathlib import Path
from evaluation.native_resume_correctness_e4_phase1a import launch as frozen
if __name__=='__main__':
    frozen.OUT=Path(__file__).resolve().parent
    frozen.main()
