"""Record the actual working files, including uncommitted research changes."""
import hashlib
from pathlib import Path


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = list((root/"hwap").glob("*.py"))+list((root/"agent_g1").glob("*.py"))
    paths += list((root/"models/unitree_g1").glob("*.xml"))
    paths += [root/"models/unitree_rl_gym/motion.pt"]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)
            if p.is_file() and not p.name.startswith('._')}
