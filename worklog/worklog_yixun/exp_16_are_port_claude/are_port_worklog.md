# Lab notebook — exp_16_are_port
## 2026-08-14 — scaffold (Yixun: "First finish a and then run ARE-on-FLAC")
- plan drafted; Codex plan review launched; GPU ordering question posed to Yixun (co-tenant vs after-DS-PA vs displace-DS-CS3). Nothing launched.

## 2026-08-14T11:13:19-04:00 — Yixun directives consolidated
- "I need ARE-V training to be done by Aug 16" + "First you just concern about ARE-V training… after ARE-V training, please remind me of the ablation experiment of ARE-FA and ARE-CYL training, and then let's discuss how to do them together."
- **Scope now: ARE-V ONLY.** Phases 2/3 (ARE-FA, ARE-CYL) deferred; a REMINDER + scheduling options go into the ARE-V completion report (due ~8/16 16:00). The cyl-backbone question (in-repo cyl_vit vs sibling cylindrical-dinov3 + weights) remains OPEN for that discussion.
- **DS-PA pause interpretation:** Yixun did not answer the pause question directly; "only ARE-V + Aug-16 deadline" admits no arithmetic without the pause, so the pause at a ckpt boundary tonight ~20:00 is treated as authorized, WITH a stated veto window until then. Logged as an interpretation, not a verbatim instruction.

## 2026-08-14T14:03:55-04:00 — Yixun HOLD directive (verbatim): "could you please stop exp_14 DS-PA by the end of 5000 step checkpoint? and please hold the exp_14 DS-CS3 and exp_16 ARE-V not to be run."
- DS-PA: pause-at-5000 watcher already armed (unchanged). DS-CS3: stays frozen. **ARE-V: tonight's probe/stamp/FULL launch CANCELLED; code stays committed + reviewed, launch-ready on his word.**
- **Consequence stated:** the Aug-16 22:00 evaluation deadline is UNREACHABLE while the hold stands (training needs ~46 h from go). Reactivation = one instruction; probe+launch within ~1 h of it.
