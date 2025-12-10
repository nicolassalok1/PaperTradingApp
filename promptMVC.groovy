You are Codex operating at the root of the PaperTradingApp repository.

GOAL  
Run ALL scripts inside `scripts/` automatically, detect ANY Python error (ImportError, ModuleNotFoundError, circular import, MVC violation, bad path, missing __init__, wrong relative import, Streamlit imported in model, etc.), and FIX them automatically.

This is an *autofix toolchain* for verifying the MVC integrity of the repository.

========================================================
GLOBAL RULES
========================================================
1. The project follows strict MVC:
   - app/model/** contains ONLY domain logic and API/business-level code.
   - app/controller/** contains ONLY routing between UI and model.
   - app/vue/** contains ONLY Streamlit UI code.
   - app/services/** contains orchestration helpers (but NOT UI).
   - app/utils/** contains shared helpers.

2. MODEL MUST NEVER import:
      streamlit, st.*, app.vue.*
3. CONTROLLER may NOT import:
      app.vue.*
4. VIEW (app/vue) MUST NOT import:
      app.model.* internal classes directly (only through controllers).

5. All modules must be importable via:
      python -m py_compile <file>

6. Every script in `scripts/*.py` must run with:
      python scripts/<script>.py

========================================================
TASK A — DISCOVER SCRIPTS
========================================================
Enumerate every file matching:
    scripts/*.py

Build a list: SCRIPTS_TO_RUN

========================================================
TASK B — EXECUTE ALL SCRIPTS
========================================================
For each script in SCRIPTS_TO_RUN:

    1. Run:
         python scripts/<script>.py
    2. Capture stdout, stderr.
    3. If ANY exception occurs (ImportError, ModuleNotFoundError,
       AttributeError, circular imports, missing modules, wrong paths,
       MVC violations):
          → STOP
          → Parse error message
          → Apply an automatic FIX to the repository.

FIX types include:

- Adding missing __init__.py
- Fixing incorrect imports:
       from app.model.X import Y   vs   from app.model.x import Y
- Fixing circular dependencies by moving imports inside functions
- Rewriting paths:
       import model.foo → from app.model.foo import ...
- Enforcing MVC:
       If a model file imports streamlit → REMOVE and move logic to controller/tab
       If a controller imports app.vue.* → REMOVE and adapt
- Normalize all model imports to:
       from app.model.<module> import <names>
- Normalize all controller imports to:
       from app.controller.<module> import <names>
- Normalize all view imports to:
       from app.controller.<module> import <names>

After each fix:
    Re-run the failing script automatically.

Repeat until:
    - All scripts run without errors, OR
    - No further fix is possible.

========================================================
TASK C — FULL-REPO IMPORT CHECK
========================================================
Before and after the fixes, run:

    python -m py_compile $(git ls-files '*.py')

For any file that fails:
    - Autofix incorrect import path
    - Add missing __all__ or __init__
    - Remove UI/MVC violations
    - Fix relative vs absolute imports
    - Then re-run py_compile

========================================================
TASK D — FINAL REPORT
========================================================

When all scripts run cleanly AND the repository compiles,
print:

    "ALL SCRIPTS PASSED. MVC CLEAN. IMPORTS CLEAN. NO ERRORS REMAIN."

If any scripts cannot be fixed, print:
    - script name
    - exact error
    - recommended manual fix.

========================================================
EXECUTION
========================================================

Now execute the FULL autofix procedure:

1. Discover scripts in ./scripts/*.py  
2. Run each script  
3. Autofix all import errors and MVC violations  
4. Repeat until stable  
5. Print final report

EXECUTE NOW.