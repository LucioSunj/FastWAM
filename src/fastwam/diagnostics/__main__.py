"""Warning-free CLI entry point: ``python -m fastwam.diagnostics``.

``python -m fastwam.diagnostics.probe`` also works, but because `__init__`
re-exports from `probe`, runpy emits a "found in sys.modules ... may result in
unpredictable behaviour" warning on stderr. Harmless -- `main()` keeps no
module-level state and the JSON still goes to stdout -- but noisy for a tool whose
output is meant to be piped, so prefer this form.
"""

from .probe import main

if __name__ == "__main__":
    raise SystemExit(main())
