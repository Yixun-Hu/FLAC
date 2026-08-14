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

## 2026-08-14T15:02:50-04:00 — Yixun: "ARE-V先不做" — HOLD CONFIRMED, deadline formally dropped
- ARE-V stays unlaunched. The Aug-16 22:00 evaluation deadline he set earlier is now MOOT by his own instruction (the last launch window, tonight 23:00, will pass unused). No further deadline is tracked for exp_16.
- **State parked launch-ready:** code committed (`6956cbc` r1, `50679ec` r2: 193 ARE tests, suite 1472 green, 76/76 guards, live mutation evidence); r2 fix-verify review in flight and will be archived; calibration done (δ̂=0, A_g=0.394574); arm config + launcher + evidence machinery complete. Reactivation cost from a cold start: probe + stamp + launch ≈ 1 h, then ~46 h training.
- **Owed when it resumes:** SOP artifacts not yet written — `are_port_params_set_up.md` exists, still missing `are_port_command.md` (beyond the stub), `commits_are_port.md`, and the results/analysis/HTML/closure set.
