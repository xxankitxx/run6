#!/usr/bin/env python3
"""
HAWK TRADER — Launcher
Run this file: python hawk_run.py
"""
import multiprocessing as mp
import os, sys, time

if __name__ == "__main__":
    mp.freeze_support()

    # Pre-import breeze_connect here in __main__ so it's cached in sys.modules.
    # hawk_trader.py will reuse the cached module — no second download.
    print("  Loading ICICI connection...", end="", flush=True)
    t0 = time.time()
    try:
        from breeze_connect import BreezeConnect
        print(f" done ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"\n  FATAL: breeze_connect failed: {e}")
        sys.exit(1)

    # Now import and run hawk_trader — BreezeConnect already in sys.modules,
    # so hawk_trader's `from breeze_connect import BreezeConnect` is instant.
    from hawk_trader import main
    main()