# Backward-compatible entry point for existing local build commands.
from pathlib import Path
exec(compile((Path(SPECPATH)/'planetrecon.spec').read_text(), 'planetrecon.spec', 'exec'))
