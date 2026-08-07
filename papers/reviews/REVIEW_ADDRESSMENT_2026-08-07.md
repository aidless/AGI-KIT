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
| R1/R2/R3: GAIA2-mini ablation is saturated and non-discriminative | Section 4.1 | Added exact 95% CI [40.0%, 97.2%] for 7/9 and sample-size guidance (415 tasks/arm for a 10 pp effect at 80% power), making "uninformative" quantitative rather than impressionistic |
| R1/R2/R3: continual learning never shows an accepted update | Section 5.3, Section 9, Section 8.4 | Added a five-criterion acceptance protocol (executable via deployment path, frozen held-out eval, statistical threshold, gate accept, audited swap); Round 18 and Round 20 failures are mapped to the criteria explicitly |
| R1/R2: small samples and wide variance make significance hard to judge | Section 3.3, Section 4.5, Sections 4.1.2 / 7.5 / 8.4 | Added exact Clopper-Pearson intervals for every inferential proportion and stated conventions; one-sample intervals are distinguished from paired tests |
| R2/R3: reproducibility gaps (prompt comparison, repeated runs) | Appendix B | Added a provenance ledger marking seed control, gold tagging, and runner preservation per experiment; Round 16's missing runner is flagged, not hidden |
| R3: safety-gate stress test is small | Section 4.5, Section 6.2, Section 6.5 | Reported 12/12 with exact CI [73.5%, 100%] and the 30-case schema-mutation test (18/18 blocked, 12/12 controls); limitation remains explicit |
| R3: impact story is a system demonstration | Section 10, Section 11 | Positioning left deliberately modest; the paper does not claim architectural priority or broad generalization |

**Still open (require new evidence, not text):** canonical GAIA2 end-to-end
scoring; an L3 candidate that completes all five acceptance criteria;
architecture-level baselines (Voyager/MetaGPT/Reflexion) on identical
hardware; larger cross-model sweeps.
