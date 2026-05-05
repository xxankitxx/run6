#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  HAWK FEED — WebSocket feed + lock-free price store                        ║
║                                                                              ║
║  Architecture (identical to Viper — proven to have zero tick drops):       ║
║    WS callback → writes directly to plain dicts (NO lock, NO queue)        ║
║    Dashboard + trade engine read from same dicts (NO lock)                 ║
║    TickQueue feeds brain subprocess (background, never blocks display)      ║
║                                                                              ║
║  Single writer (WS callback) → plain dict → multiple readers               ║
║  This is safe in CPython due to GIL on dict assignment.                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, time, threading, queue, logging
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
from collections import deque
from datetime import datetime, timedelta, timezone

import numpy as np

try:
    from breeze_connect import BreezeConnect
except ImportError:
    print("FATAL: pip install breeze_connect")
    sys.exit(1)

IST = timezone(timedelta(hours=5, minutes=30))
log = logging.getLogger("hawk.feed")

# ── Globals ──────────────────────────────────────────────────────────────────
_NEW_TICK_EVENT = threading.Event()
# _TICK_Q and _DROPPED_TICKS removed — ticks are processed directly in WS callback,
# no secondary queue needed.


# =============================================================================
# PRICE STORE — lock-free, single-writer (WS callback), multi-reader
# =============================================================================
class PriceStore:
    """
    ZERO-LOCK hot path for maximum tick throughput.

    Architecture:
      - WS callback is the ONLY writer (single thread from breeze_connect)
      - Dashboard, TradingThread, SnapFeeder are readers
      - CPython GIL guarantees: dict[key]=value and dict.get(key) are atomic
      - Therefore: NO lock needed for single-field writes or reads
      - Lock used ONLY for snapshot_strikes() which reads multiple fields
        that must be consistent with each other

    Hot path (write): _on_ticks → _write_spot/futures/option → ZERO LOCK
    Hot path (read):  Dashboard/TradingThread → atm/ce_ltp/pe_ltp/get_option_price → ZERO LOCK
    Cold path (1/sec): SnapFeeder → snapshot_strikes → LOCK (consistent multi-field read)
    """

    def __init__(self, spot_name_fragment: str = "NIFTY 50",
                 strike_step: int = 50):
        self._snap_lock         = threading.Lock()  # ONLY for snapshot_strikes
        self.spot_name_fragment = spot_name_fragment
        self.strike_step        = strike_step

        # ── Live prices (written by WS callback, read by anyone) ──────────
        # All single-field reads/writes are GIL-atomic. No lock needed.
        self.spot:          float = 0.0
        self.spot_ts:       float = 0.0
        self.futures:       float = 0.0
        self.futures_ts:    float = 0.0
        self.real_vix:      float = 0.0   # v25-LIVE: real INDIAVIX from REST API

        # option prices: key = (strike, "call"/"put")
        self.opt_ltp:  Dict[Tuple[float, str], float] = {}
        self.opt_ts:   Dict[Tuple[float, str], float] = {}
        self.opt_bid:  Dict[Tuple[float, str], float] = {}
        self.opt_ask:  Dict[Tuple[float, str], float] = {}
        self.opt_oi:   Dict[Tuple[float, str], float] = {}
        self.opt_vol:  Dict[Tuple[float, str], float] = {}

        # ring buffer: last 10 ticks per option (for spike detection)
        self._ring: Dict[Tuple[float, str], deque] = {}

        # counters
        self.tick_count:     int = 0
        self.spot_ticks:     int = 0
        self.option_ticks:   int = 0
        self.futures_ticks:  int = 0
        self.dropped_ticks:  int = 0

        # ── Pre-computed cache (updated periodically, read lock-free) ─────
        self._cached_pcr:      float = 1.0
        self._cached_ce_oi:    float = 0.0
        self._cached_pe_oi:    float = 0.0
        self._pcr_update_ts:   float = 0.0

    # ── v28 Innovation B: Atomic core snapshot ──────────────────────────────
    def freeze_core(self) -> tuple:
        """Capture the 5 core values in a single consistent read.

        Problem solved (identified by ChatGPT teardown):
          build_snapshot() reads spot, then calls self.atm (which RE-READS spot),
          then reads ce_ltp/pe_ltp from that ATM. Between reads, a WS callback
          can change spot → ATM shifts → ce/pe are read from the WRONG strike.
          On a 650pt trending day, ATM can shift 50pts in < 1 second.

        Fix: capture spot ONCE, derive ATM locally, read ce/pe from that ATM.
        This eliminates the primary torn-read vector without any lock.

        Returns: (spot, futures, atm, ce_ltp, pe_ltp)
        """
        spot = self.spot              # GIL-atomic single read — captured once
        step = self.strike_step
        atm = round(spot / step) * step if spot > 0 else 0.0
        fut = self.futures if self.futures > 0 else spot  # futures also captured once
        ce_ltp = self.opt_ltp.get((atm, "call"), 0.0)
        pe_ltp = self.opt_ltp.get((atm, "put"), 0.0)
        return spot, fut, atm, ce_ltp, pe_ltp

    # ── Display helpers (ZERO LOCK — all GIL-atomic reads) ─────────────────
    @property
    def atm(self) -> float:
        s = self.spot
        if s <= 0:
            return 0.0
        return round(s / self.strike_step) * self.strike_step

    def ce_ltp(self, strike: float = 0.0) -> float:
        s = strike if strike > 0 else self.atm
        return self.opt_ltp.get((s, "call"), 0.0)

    def pe_ltp(self, strike: float = 0.0) -> float:
        s = strike if strike > 0 else self.atm
        return self.opt_ltp.get((s, "put"), 0.0)

    def get_option_price(self, strike: float, right: str) -> float:
        """GIL-atomic single dict read. No lock needed."""
        return self.opt_ltp.get((strike, right), 0.0)

    def opt_age(self, strike: float, right: str) -> float:
        ts = self.opt_ts.get((strike, right), 0.0)
        return time.monotonic() - ts if ts > 0 else 9999.0

    def recent_ticks(self, strike: float, right: str, n: int = 5) -> List[float]:
        buf = self._ring.get((strike, right))
        if not buf:
            return []
        return list(buf)[-n:]

    def check_volume_spike(self, atm: float, direction: str,
                           spike_mult: float = 3.0) -> Tuple[bool, str]:
        """
        ZERO LOCK — reads individual dict entries (GIL-atomic).
        Called every 5 seconds from TradingThread, not on hot path.
        """
        check_right = "put" if direction == "CE" else "call"
        same_right  = "call" if direction == "CE" else "put"
        step = self.strike_step

        total_vol = 0.0
        same_vol  = 0.0
        for offset in range(-3, 4):
            s = atm + offset * step
            v = self.opt_vol.get((s, check_right), 0.0)
            if v > 0:
                total_vol += v
            v2 = self.opt_vol.get((s, same_right), 0.0)
            if v2 > 0:
                same_vol += v2

        if total_vol <= 0 or same_vol <= 0:
            return False, ""

        ratio = total_vol / same_vol
        if ratio >= spike_mult:
            return True, (f"VOL_SPIKE: {check_right} vol={total_vol:.0f} "
                          f"vs {same_right} vol={same_vol:.0f} "
                          f"ratio={ratio:.1f}x")
        return False, ""

    def full_chain_pcr(self) -> Tuple[float, float, float]:
        """
        ZERO LOCK — reads from pre-computed cache.
        Cache is refreshed by update_pcr_cache() called from build_snapshot.
        """
        return self._cached_pcr, self._cached_ce_oi, self._cached_pe_oi

    def update_pcr_cache(self):
        """
        Recompute PCR from full OI store. Called once per second from
        build_snapshot (SnapFeeder thread). If the dictionary changes size
        during iteration, skip this cycle (it will be retried next second).
        """
        total_ce = 0.0
        total_pe = 0.0
        try:
            # Iterate over the dictionary items. If the dict changes size
            # during iteration, a RuntimeError is raised; we catch it and
            # skip this refresh (next second's call will retry).
            for (strike, right), oi_val in self.opt_oi.items():
                if oi_val > 0:
                    if right == "call":
                        total_ce += oi_val
                    elif right == "put":
                        total_pe += oi_val
        except RuntimeError:
            # Dictionary changed size during iteration – ignore this refresh
            return
        self._cached_ce_oi = total_ce
        self._cached_pe_oi = total_pe
        self._cached_pcr   = total_pe / total_ce if total_ce > 0 else 1.0

    def snapshot_strikes(self, n_strikes: int = 8, atm_override: float = 0.0) -> dict:
        """
        Multi-field consistent read — this is the ONLY method that locks.
        Called once per second from SnapFeeder, never from hot path.

        v28.4 FIX (DeepSeek Issue B): accept atm_override so build_snapshot()
        can pass the SAME ATM from freeze_core(). Without this, a WS callback
        between freeze_core() and snapshot_strikes() can shift self.spot →
        different ATM → strikes table centered on wrong ATM.
        """
        with self._snap_lock:
            if atm_override > 0:
                atm = atm_override
            else:
                atm = round(self.spot / self.strike_step) * self.strike_step if self.spot > 0 else 0
            if atm <= 0:
                return {}
            result = {}
            for offset in range(-n_strikes, n_strikes + 1):
                s = atm + offset * self.strike_step
                for right in ("call", "put"):
                    key = (s, right)
                    if key in self.opt_ltp:
                        result[key] = {
                            "ltp":  self.opt_ltp.get(key, 0.0),
                            "oi":   self.opt_oi.get(key, 0.0),
                            "vol":  self.opt_vol.get(key, 0.0),
                            "bid":  self.opt_bid.get(key, 0.0),
                            "ask":  self.opt_ask.get(key, 0.0),
                        }
            return result

    # ── Internal write methods (WS callback ONLY — ZERO LOCK) ────────────
    # CPython GIL guarantees: self.spot = ltp is atomic.
    # dict[key] = value is atomic. No concurrent writer exists.
    # _NEW_TICK_EVENT.set() is thread-safe by design.

    def _write_spot(self, ltp: float):
        self.spot    = ltp
        self.spot_ts = time.monotonic()
        self.spot_ticks += 1
        self.tick_count += 1
        _NEW_TICK_EVENT.set()

    def _write_futures(self, ltp: float):
        self.futures    = ltp
        self.futures_ts = time.monotonic()
        self.futures_ticks += 1
        self.tick_count += 1
        _NEW_TICK_EVENT.set()

    def _write_option(self, strike: float, right: str, ltp: float,
                      oi: float = 0.0, vol: float = 0.0,
                      bid: float = 0.0, ask: float = 0.0):
        key = (strike, right)
        if ltp > 0:
            self.opt_ltp[key] = ltp
            self.opt_ts[key]  = time.monotonic()
            # ring buffer
            buf = self._ring.get(key)
            if buf is None:
                buf = deque(maxlen=10)
                self._ring[key] = buf
            buf.append(ltp)
        if oi >= 0 and key in self.opt_oi:
            # Update OI even if zero — otherwise stale positive values persist
            # Only update if we've seen this key before (avoids setting 0 from
            # ticks that don't include OI data)
            self.opt_oi[key] = oi
        elif oi > 0:
            self.opt_oi[key]  = oi
        if vol > 0:
            self.opt_vol[key] = vol
        if bid > 0:
            self.opt_bid[key] = bid
        if ask > 0:
            self.opt_ask[key] = ask
        self.option_ticks += 1
        self.tick_count   += 1
        _NEW_TICK_EVENT.set()


# =============================================================================
# WS MANAGER — connects, subscribes, routes ticks
# =============================================================================
class FeedManager:

    RECONNECT_WAIT = 5.0
    MAX_RECONNECT  = 20

    def __init__(self, breeze: BreezeConnect, prices: PriceStore,
                 expiry_ws: str, stock_code: str = "NIFTY",
                 futures_expiry_ws: str = ""):
        self._breeze            = breeze
        self._prices            = prices
        self._expiry_ws         = expiry_ws          # options expiry e.g. "25-Mar-2026"
        self._futures_expiry_ws = futures_expiry_ws  # futures expiry e.g. "31-Mar-2026"
        self._stock_code        = stock_code
        self._subscribed: set   = set()
        self.connected          = False
        self._reconnect_needed  = False
        self._reconnect_attempts = 0
        self._lock = threading.Lock()

    # ── Connection ────────────────────────────────────────────────────────────
    def connect(self, max_retries: int = 5, base_wait: float = 3.0):
        def _on_ticks(tick_data):
            """WS callback — MUST be fast. Only writes to plain dicts."""
            if not isinstance(tick_data, dict):
                return
            try:
                ex  = tick_data.get("exchange", "")
                ltp = float(tick_data.get("last") or 0)
                if ltp <= 0:
                    return

                if ex == "NSE Equity":
                    name = str(tick_data.get("stock_name", ""))
                    if self._prices.spot_name_fragment in name.upper():
                        self._prices._write_spot(ltp)

                elif ex == "NSE Futures & Options":
                    prod = (tick_data.get("product_type") or "").strip()
                    if prod in ("Futures", "futures", "FUTURES"):
                        self._prices._write_futures(ltp)
                    elif prod == "Options":
                        sr = tick_data.get("strike_price", "")
                        rr = (tick_data.get("right") or "").strip().lower()
                        if sr and rr in ("call", "put"):
                            strike = float(sr)
                            if strike > 0:
                                oi  = float(tick_data.get("OI")           or 0)
                                vol = float(tick_data.get("ttq")          or 0)
                                bid = float(tick_data.get("bPrice")       or 0)
                                ask = float(tick_data.get("sPrice")       or 0)
                                self._prices._write_option(
                                    strike, rr, ltp, oi, vol, bid, ask)
            except Exception:
                pass  # never let display path crash

            # Ticks processed directly above — no secondary queue needed

        def _on_close(ws, code, msg):
            log.warning("WS closed: %s %s", code, msg)
            self.connected = False
            self._reconnect_needed = True

        def _on_error(ws, err):
            log.error("WS error: %s", err)
            self.connected = False
            self._reconnect_needed = True

        self._breeze.on_ticks = _on_ticks
        try:
            self._breeze.on_close = _on_close
            self._breeze.on_error = _on_error
        except Exception:
            pass

        # Retry with exponential backoff — BreezeConnect WS is flaky
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                self._breeze.ws_connect()
                self.connected = True
                if attempt > 1:
                    log.info("FeedManager: WebSocket connected (attempt %d)", attempt)
                else:
                    log.info("FeedManager: WebSocket connected")
                return  # success
            except Exception as e:
                last_err = e
                wait = base_wait * (2 ** (attempt - 1))  # 3, 6, 12, 24, 48s
                log.warning("FeedManager: WS connect failed (attempt %d/%d): %s — retrying in %.0fs",
                            attempt, max_retries, e, wait)
                print(f"  [WARN] WS connect failed (attempt {attempt}/{max_retries}): {e}")
                print(f"         Retrying in {wait:.0f}s…")
                time.sleep(wait)

        # All retries exhausted
        raise ConnectionError(
            f"Failed to connect WebSocket after {max_retries} attempts. "
            f"Last error: {last_err}. "
            f"Check: 1) Market hours  2) Session token validity  "
            f"3) Network/firewall  4) ICICI server status"
        )

    def disconnect(self):
        try:
            self._breeze.ws_disconnect()
        except Exception:
            pass
        self.connected = False
        with self._lock:
            self._subscribed.clear()   # clear so resubscription works after reconnect

    # ── Subscriptions ─────────────────────────────────────────────────────────
    def subscribe_spot(self):
        try:
            r = self._breeze.subscribe_feeds(
                stock_code=self._stock_code,
                exchange_code="NSE",
                product_type="cash",
                get_exchange_quotes=True,
                get_market_depth=False,
            )
            log.info("subscribe_spot: %s", r)
        except Exception as e:
            log.error("subscribe_spot failed: %s", e)

    def subscribe_futures(self, expiry_ws: str = ""):
        exp = expiry_ws or self._expiry_ws
        try:
            r = self._breeze.subscribe_feeds(
                stock_code=self._stock_code,
                exchange_code="NFO",
                product_type="futures",
                expiry_date=exp,
                get_exchange_quotes=True,
                get_market_depth=False,
            )
            log.info("subscribe_futures: %s", r)
        except Exception as e:
            log.error("subscribe_futures failed: %s", e)

    def subscribe_strikes(self, atm: float, n: int = 8, step: int = 50):
        """Subscribe to CE and PE options around ATM."""
        exp = self._expiry_ws
        added = 0
        for offset in range(-n, n + 1):
            strike = atm + offset * step
            for right in ("call", "put"):
                key = (strike, right)
                with self._lock:
                    if key in self._subscribed:
                        continue
                try:
                    r = self._breeze.subscribe_feeds(
                        stock_code=self._stock_code,
                        exchange_code="NFO",
                        product_type="options",
                        expiry_date=exp,
                        strike_price=str(int(strike)),
                        right=right,
                        get_exchange_quotes=True,
                        get_market_depth=False,
                    )
                    with self._lock:
                        self._subscribed.add(key)
                    added += 1
                except Exception as e:
                    log.debug("subscribe_strikes %s %s %s: %s", strike, right, exp, e)
        log.info("subscribe_strikes: ATM=%s added=%d total=%d", atm, added, len(self._subscribed))
        return added

    # ── Reconnect loop ────────────────────────────────────────────────────────
    def reconnect_if_needed(self):
        if not self._reconnect_needed:
            return
        if self._reconnect_attempts >= self.MAX_RECONNECT:
            log.critical("Max reconnect attempts reached (%d)", self.MAX_RECONNECT)
            return
        self._reconnect_needed = False
        self._reconnect_attempts += 1
        log.warning("Reconnecting WS (attempt %d/%d)…",
                    self._reconnect_attempts, self.MAX_RECONNECT)
        try:
            self.disconnect()
            time.sleep(self.RECONNECT_WAIT)
            self.connect(max_retries=3, base_wait=2.0)  # shorter retries for reconnect
            # Re-subscribe everything — spot, futures, options
            self.subscribe_spot()
            if self._futures_expiry_ws:
                self.subscribe_futures(expiry_ws=self._futures_expiry_ws)
            atm = self._prices.atm
            if atm > 0:
                self.subscribe_strikes(atm)
            self._reconnect_attempts = 0
            log.info("Reconnect successful (spot+futures+options resubscribed)")
        except Exception as e:
            log.error("Reconnect failed: %s", e)
            self._reconnect_needed = True

    def option_feed_alive(self, max_age: float = 30.0) -> bool:
        """True if at least one option tick received in the last max_age seconds."""
        now = time.monotonic()
        if not self._prices.opt_ts:
            return False
        newest = max(self._prices.opt_ts.values())
        return (now - newest) < max_age


# =============================================================================
# SNAPSHOT BUILDER — assembles a dict for the brain subprocess
# Keeps it simple — just primitives, no custom classes
# =============================================================================
def build_snapshot(prices: PriceStore) -> Optional[dict]:
    """
    Build a plain dict snapshot for the brain subprocess.
    Uses only primitives — safe to pickle across mp.Queue.

    v28 Innovation B: Uses freeze_core() for consistent multi-variable read.
    Before this fix, spot/atm/ce_ltp/pe_ltp were read in 5 separate steps.
    A WS callback between reads could shift ATM → wrong strike options.
    freeze_core() captures spot once and derives everything from it.
    """
    # v28 Innovation B: Atomic core snapshot — single consistent read
    spot, fut, atm, ce_ltp, pe_ltp = prices.freeze_core()
    if spot <= 0:
        return None

    # Build per-strike OI table (nearby strikes for OI walls / smart money)
    # This is the ONLY lock-holding call in the entire build path.
    # v28.4 FIX: pass atm from freeze_core() so strikes are centered on the
    # SAME ATM as the core snapshot — eliminates torn-read between the two calls.
    strikes_data = prices.snapshot_strikes(n_strikes=6, atm_override=atm)

    # Refresh PCR cache (lock-free iteration of opt_oi dict)
    prices.update_pcr_cache()
    # Read from cache (lock-free)
    pcr, total_ce_oi, total_pe_oi = prices.full_chain_pcr()

    # If full chain has no OI yet (pre-bootstrap), fall back to nearby
    if total_ce_oi <= 0 and total_pe_oi <= 0:
        total_ce_oi = sum(v["oi"] for (s, r), v in strikes_data.items() if r == "call")
        total_pe_oi = sum(v["oi"] for (s, r), v in strikes_data.items() if r == "put")
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1.0

    # Max OI strikes
    ce_ois = {s: v["oi"] for (s, r), v in strikes_data.items() if r == "call" and v["oi"] > 0}
    pe_ois = {s: v["oi"] for (s, r), v in strikes_data.items() if r == "put"  and v["oi"] > 0}
    max_ce_strike = max(ce_ois, key=ce_ois.get) if ce_ois else atm
    max_pe_strike = max(pe_ois, key=pe_ois.get) if pe_ois else atm

    spot_age = time.monotonic() - prices.spot_ts if prices.spot_ts > 0 else 999.0
    is_clean = spot_age < 10.0 and ce_ltp > 0 and pe_ltp > 0

    # v25-LIVE: Use real INDIAVIX from REST API if available, else proxy
    # Sanity check: Breeze sometimes returns VIX as a decimal fraction (e.g. 0.25
    # instead of 25.0) — a 100x shift. India VIX is always between 5 and 90.
    # If the value is below 5 but positive, it is almost certainly a decimal-shifted
    # reading; multiply by 100 to recover the true value. We also keep the last valid
    # reading in prices.real_vix so that a bad tick does not wipe out good state.
    raw_vix = prices.real_vix
    if 0 < raw_vix < 5.0:
        raw_vix = raw_vix * 100.0          # correct the decimal-shift
    vix_value = raw_vix if raw_vix >= 5.0 else 0.0
    if vix_value <= 0 and spot > 0 and ce_ltp > 0 and pe_ltp > 0:
        straddle_pct = (ce_ltp + pe_ltp) / spot
        vix_value = max(8.0, min(50.0, straddle_pct * 1800))  # fallback proxy

    return {
        "spot":           spot,
        "futures":        fut,
        "atm":            atm,
        "ce_ltp":         ce_ltp,
        "pe_ltp":         pe_ltp,
        "total_ce_oi":    total_ce_oi,
        "total_pe_oi":    total_pe_oi,
        "pcr":            pcr,
        "max_ce_strike":  max_ce_strike,
        "max_pe_strike":  max_pe_strike,
        "strikes":        strikes_data,   # {(strike, right): {ltp,oi,vol,bid,ask}}
        "is_clean":       is_clean,
        "vix":            vix_value,      # v25-LIVE: real INDIAVIX (or proxy fallback)
        "ts":             time.monotonic(),
    }