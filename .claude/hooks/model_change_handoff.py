#!/usr/bin/env python3
"""
model_change_handoff.py - Claude Code hook (wired on SessionStart + UserPromptSubmit).

Detects a main-session model change and:
  (a) snapshots the four handoff docs into worklog/worklog_yixun/handoff_snapshots/<ts>/,
  (b) appends a row to worklog/worklog_yixun/handoff_log.md,
  (c) injects a reminder via hookSpecificOutput.additionalContext so the incoming model
      refreshes the handoff docs per the CLAUDE.md "Session handoff & compaction protocol".

The hook is the DETECTOR/ARCHIVER only; the live model still authors the doc refresh.

Fail-safe: any error, non-dict payload, or undetermined model is a silent no-op (exit 0);
the hook can never block or corrupt a turn.

Detection:
  - SessionStart carries a top-level "model"; other events do not, so we read the last
    NON-sidechain assistant record's message.model from the transcript tail (bounded).
  - Synthetic/error records (model beginning with "<", e.g. "<synthetic>") are ignored --
    they would otherwise toggle real->synthetic->real and spam reminders.
  - Identity is compared at the model-FAMILY level (fable/opus/sonnet/haiku). One model shows
    up under several spellings across event sources (SessionStart said "claude-opus-4-7" while
    the transcript says "claude-opus-4-8" and "opus"); comparing those verbatim would ping-pong
    the marker and spam. Family granularity fires on a real switch (fable->opus, the user's
    stated case) but not on intra-family version/spelling/effort/context flips.
  - The marker is SESSION-SCOPED (.last_model__<session_id>) and the read/compare/write runs
    under an exclusive flock, so concurrent sessions using different models never toggle a
    shared marker. Snapshot+log happen BEFORE the marker is committed (atomic replace), so a
    mid-transaction failure re-fires (at-least-once) rather than silently losing the change.
  - LIMITATION: on UserPromptSubmit the transcript's last assistant record is the PREVIOUS
    turn's model, so a mid-session /model switch is detected one prompt late. SessionStart is
    authoritative and immediate; it is the only event that currently exposes the model directly.
"""
import sys, os, json, re, shutil, datetime, fcntl

_FAMILIES = ("fable", "opus", "sonnet", "haiku")


def _family(m):
    # Coarse identity: the family token, robust to spelling/version/effort/context variation.
    s = str(m or "").strip().lower()
    for fam in _FAMILIES:
        if fam in s:
            return fam
    return re.sub(r"\s*\[[^\]]*\]\s*$", "", s).strip(" -_")  # unknown family: normalized full id


def _is_real_model(m):
    m = str(m or "").strip()
    return bool(m) and not m.startswith("<")  # reject <synthetic>, <error>, ...


def _model_from_transcript(path, max_bytes=4_000_000):
    if not path or not os.path.isfile(path):
        return ""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()  # discard a partial first line
            data = f.read()
        for line in reversed(data.decode("utf-8", "ignore").splitlines()):
            line = line.strip()
            if not line or '"model"' not in line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict) or obj.get("type") != "assistant" or obj.get("isSidechain"):
                continue
            m = (obj.get("message") or {}).get("model")
            if _is_real_model(m):
                return str(m).strip()
    except Exception:
        return ""
    return ""


def _safe(s):
    return "".join(c if (c.isalnum() or c in ".-_") else "-" for c in str(s))[:64]


def _append_log(hb, ts, prev, cur, event, snap_rel):
    try:
        p = os.path.join(hb, "handoff_log.md")
        fresh = not os.path.isfile(p)
        with open(p, "a") as f:
            if fresh:
                f.write("# Handoff log - model changes\n\nAuto-appended by "
                        "`.claude/hooks/model_change_handoff.py`. The hook archives the four handoff docs "
                        "and reminds the incoming model to refresh them; the live model authors the refresh.\n\n"
                        "| timestamp | event | from | to | snapshot |\n|---|---|---|---|---|\n")
            f.write(f"| {ts} | {event} | {prev or '(none)'} | {cur} | {snap_rel or '-'} |\n")
    except Exception:
        pass


def _run(data):
    if not isinstance(data, dict):
        return
    event = str(data.get("hook_event_name") or "")
    transcript = data.get("transcript_path") or ""
    cwd = data.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    sid = _safe(data.get("session_id") or "global")

    cur_raw = str(data.get("model") or "").strip()
    if not _is_real_model(cur_raw):
        cur_raw = _model_from_transcript(transcript)
    cur = _family(cur_raw)
    if not cur:
        return

    hb = os.path.join(cwd, "worklog", "worklog_yixun")
    if not os.path.isdir(hb):
        return
    marker = os.path.join(hb, f".last_model__{sid}")

    lockf = open(os.path.join(hb, ".handoff_hook.lock"), "w")
    try:
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            return  # another hook holds the lock; let it handle the change
        prev = ""
        if os.path.isfile(marker):
            try:
                prev = open(marker).read().strip()
            except Exception:
                prev = ""
        if prev == cur:
            return

        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        n = 0
        snap_rel = ""
        if prev:  # real change: snapshot + log BEFORE committing the marker
            docs = ["CLAUDE.md",
                    "worklog/worklog_yixun/master_experiment_tracker.md",
                    "worklog/worklog_yixun/issue_report.md",
                    "worklog/worklog_yixun/HANDOFF.md"]
            snapdir = os.path.join(hb, "handoff_snapshots", f"{ts}__{_safe(prev)}__to__{_safe(cur)}")
            try:
                os.makedirs(snapdir, exist_ok=True)
                for d in docs:
                    src = os.path.join(cwd, d)
                    if os.path.isfile(src):
                        shutil.copy2(src, os.path.join(snapdir, os.path.basename(d)))
                        n += 1
                snap_rel = os.path.relpath(snapdir, cwd)
            except Exception:
                snap_rel = ""
            _append_log(hb, ts, prev, cur_raw or cur, event or "?", snap_rel)

        try:  # atomically commit the marker LAST (mid-transaction failure re-fires, not loses)
            tmp = marker + ".tmp"
            with open(tmp, "w") as f:
                f.write(cur)
            os.replace(tmp, marker)
        except Exception:
            return

        if not prev:
            _append_log(hb, ts, "", cur_raw or cur, event or "init", "")
            return  # first observation for this session: seed silently, no reminder

        reminder = (
            f"[handoff-hook] MAIN-SESSION MODEL CHANGED: {prev} -> {cur_raw or cur} (event={event or '?'}). "
            f"Per the CLAUDE.md 'Session handoff & compaction protocol', BEFORE other work you MUST refresh the "
            f"four handoff docs to current working memory: CLAUDE.md, "
            f"worklog/worklog_yixun/master_experiment_tracker.md, worklog/worklog_yixun/issue_report.md, and "
            f"worklog/worklog_yixun/HANDOFF.md. Archived {n}/4 to {snap_rel or '(unavailable)'} (archive only - "
            f"you author the refresh). In experiment analysis, flag that you are {cur_raw or cur}, not the prior model."
        )
        inj_event = event if event in ("SessionStart", "UserPromptSubmit") else "UserPromptSubmit"
        print(json.dumps({"hookSpecificOutput": {"hookEventName": inj_event, "additionalContext": reminder}}))
    finally:
        try:
            fcntl.flock(lockf, fcntl.LOCK_UN)
        except Exception:
            pass
        lockf.close()


def main():
    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw and raw.strip() else {}
    except Exception:
        sys.exit(0)
    try:
        _run(data)
    except Exception:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
