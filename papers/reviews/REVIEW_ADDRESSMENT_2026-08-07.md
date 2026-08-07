# Review Addressment Log (2026-08-07)

**Basis:** the TMLR-style multi-review record
(`review_preprint_unified_en_round16.json`: R1/R2/R3 all 6/10, minor
revision) and the in-repo heuristic simulator
(`papers/reviews/summary.txt`: 3.50/5, Weak Accept).

This log maps each recurring reviewer concern to the manuscript change
that responds to it. No new experiments were run in this pass; all
additions are quantitative restatements of existing artifacts or
explicit protocols for future work.

| Reviewer concern | Response in manuscript | Change |
|---|---|---|
| R1/R2/R3: headline arithmetic gain is not causally attributable to the layers | Section 4.5, Abstract, Section 8.1 | New claim-versus-evidence ledger labels every headline result controlled / configuration-level / descriptive; matched-step gap remains +60 pp with prompt and control flow flagged as confounds |
| R1/R2/R3: GAIA2-mini ablation is saturated and non-discriminative | Section 4.1 | Added exact 95% CI [40.0%, 97.2%] for 7/9 and sample-size guidance (224 tasks/arm for a 10 pp effect at 80% power, pooled-variance two-proportion formula), making "uninformative" quantitative rather than impressionistic |
| R1/R2/R3: continual learning never shows an accepted update | Section 5.3, Section 9, Section 8.4 | Added a five-criterion acceptance protocol (executable via deployment path, frozen held-out eval, statistical threshold, gate accept, audited swap); Round 18 and Round 20 failures are mapped to the criteria explicitly |
| R1/R2: small samples and wide variance make significance hard to judge | Section 3.3, Section 4.5, Sections 4.1.2 / 7.5 / 8.4 | Added exact Clopper-Pearson intervals for every inferential proportion and stated conventions; one-sample intervals are distinguished from paired tests |
| R2/R3: reproducibility gaps (prompt comparison, repeated runs) | Appendix B | Added a provenance ledger marking seed control, gold tagging, and runner preservation per experiment; Round 16's missing runner is flagged, not hidden |
| R3: safety-gate stress test is small | Section 4.5, Section 6.2, Section 6.5 | Reported 12/12 with exact CI [73.5%, 100%] and the 30-case schema-mutation test (18/18 blocked, 12/12 controls); limitation remains explicit |
| R3: impact story is a system demonstration | Section 10, Section 11 | Positioning left deliberately modest; the paper does not claim architectural priority or broad generalization |

**Still open (require new evidence, not text):** canonical GAIA2 end-to-end
scoring; an L3 candidate that completes all five acceptance criteria;
architecture-level baselines (Voyager/MetaGPT/Reflexion) on identical
hardware; larger cross-model sweeps.

## External TMLR-style review (2026-08-07, four-dimension 2.75 / Borderline)

An independent reviewer re-read the full bundle, recomputed all 11
Clopper-Pearson intervals and both McNemar p-values (all correct), and
raised three blockers plus seven suggestions. This pass addresses them:

### Blockers addressed
1. **Memory accounting** - Section 2.3 and Appendix A now give one
   per-process RSS table and one conclusion: total ~5.4 GB measured
   (3.3 GB harness + 2.1 GB Ollama) plus ~0.4 GB buffers, i.e., the
   stack exceeds the 5 GB target with both models resident. The
   earlier "~2x headroom" and "~1.7 GB headroom" claims are withdrawn.
2. **Power calculation** - Section 4.1 now states the pooled-variance
   two-proportion formula and the reproducible figures (224/arm for
   10 pp, 42/arm for 20 pp); the unexplained 415/113 figures are
   withdrawn.
3. **McNemar p-values under deterministic protocols** - Section 3.3
   adds an explicit interpretation note; both p-value sites
   (Sections 4.1.2 and 8.4) now point to it.

### Suggestions addressed
4. Table 4.1: "-4 pp vs Qwen" corrected to "-5 pp".
5. Section 5.3: the unsupported "two rejected candidates in seven that
   outperformed" sentence is rewritten using the actual 60-trial
   calibration record (at threshold 0.5 the two rejections had
   new_acc 0.1/0.3, below baseline).
6. Abstract: the 8/8 vs 77.6% comparison now states the two samples
   are not directly comparable.
7. Section 4.4: n/a entries for qwen2.5:3b / qwen3:0.6b now state that
   the full wrapper was not run, not that it failed.
8. Ledger and abstract: 1/20 (5.0%) now carries exact CI [0.1, 24.9];
   structural 20/20 carries [83.2, 100].
9. PUBLISHING.md figure list unified to Figures 1 and 5 (matching
   README and the embedded manuscript).
10. COVER_LETTER.md states the public repository URL and notes an
    anonymized mirror can be prepared if the venue requires it.

Remaining reviewer questions that need data, not text, are unchanged:
canonical GAIA2 end-to-end scoring, a fully accepted L3 candidate, and
architecture-level baselines.
