# RAVE-FL research artifact inventory (27 October 2026)

## Data source and scope

All figures below were read from the AIScalar runs provided by the researcher.
This is an **inventory and preliminary result summary**, not the complete raw
artifact backup. The original source files, request journals, result JSON,
evaluation reports, and logs live on AIScalar until a GitHub Release upload
is confirmed. Do not claim the raw artifacts are archived based only on
this Markdown commit.

Repository: `ParisaArbab/AdaptiveContextOpt`, branch
`rave-fl-adaptive-pilot`.

### Original benchmark (8K context)

- Run: `d4j97_promptfix_20261027_122406`
- 100 bugs selected; 97 passed prepare-validation; 3 excluded with
  documented source/line mismatches: `Closure-47`,
  `JxPath-7`, `JacksonDatabind-20`.
- Of 97 submitted per-bug runs: 93 exited 0 and four exited 3.
- Exit 0 is **not** equivalent to three completed policy arms or successful
  fault localization.
- The four incomplete runs: `Closure-99`, `Math-22`,
  `Codec-14`, `Lang-34`. Their Full policy hit the conservative
  prompt/context admission limit.
- Policies: `full`, `fixed_l1`, `v2_utility`.
- Model: `qwen2.5:7b` through Ollama 0.40.7.
- Context window 8,192; batch size 2; per-candidate output cap 160.
- Original logs and data:
  `results/d4j_multibug/d4j97_promptfix_20261027_122406/`
  `logs/d4j_multibug/d4j97_promptfix_20261027_122406/`
  `logs/d4j97_promptfix_20261027_122406.log`

### Controlled prompt-fix pilot

- Corrected prompt returned the correct number of candidates in all 68
  screened batches across `Chart-17`, `Cli-10`, and `Codec-16`.
- Prior prompt had 24 malformed results in 68 batches (35.3%).
- Prompt fix addresses observed response cardinality, **not** guaranteed
  accuracy improvements.
- Saved pilot run IDs:
  `chart17_promptfix_20261027_114655`,
  `pilot2_promptfix_20261027_121624`.

### Separate four-bug recovery (16K context)

- Run: `d4j_recovery4_16k_20261027_165751`
- Same three policies, model and batch size; context raised to 16,384.
- This changes the experimental conditions: report this cohort **separately**
  from the 8K results.

| Bug | Full tokens | Fixed L1 tokens | V2 tokens | Full method rank | V2 method rank | Run-level outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Closure-99 | 26,606 | 15,719 | 15,719 | 11 | 3 | Exit 0; all arms complete |
| Math-22 | 18,361 | 10,897 | 11,568 | 10 | 6 | Exit 0; all arms complete |
| Codec-14 | 16,613 | 10,845 | 11,205 | not found | not found | Exit 0; all arms complete |
| Lang-34 | 64,256 | 0 | 0 | N/A | N/A | Exit 3; all arms incomplete |

- `Lang-34` had `ExecutionLimit: Per-bug physical token cap cannot
  reserve next request`. Its two zero-token selective arms never started;
  these are **not** token savings. The token cap was 80,000 per bug
  including reservation for the next call.
- On the three completed recovery bugs, V2 used about **37.5% fewer
  tokens in aggregate than Full**. This is descriptive and based on three
  independent cases, not the whole 97-bug cohort.
- All observations were produced by real model inference; this benchmark
  measures fault localization, not bug repair or patch correctness.

## Archival and reproducibility

Use `scripts/aiscalar_package_research_outputs.sh` on AIScalar to snapshot
the complete `results/`, `logs/`, `configs/`, and, if present,
`data/rave_fl_observed_failures/` trees. It streams a gzipped tar and
splits it into <1 GB GitHub Release asset parts, plus SHA-256 checksums,
commit identity and restore instructions. This avoids force-adding
`results/` (excluded by `.gitignore`) to code commits.

**IMPORTANT: This repository is public.** Review all raw data, prompts,
credentials, internal paths, and logs before uploading. GitHub Release
assets will be public on publication. The packaging script does not
publish anything by itself.

To reassemble the archive after downloading all parts:

```bash
sha256sum -c SHA256SUMS.txt
mkdir -p restored
cat RAVE_FL_ALL_RESULTS_LOGS.tar.gz.part-* | tar -xzf - -C restored
```

The original data remain on AIScalar and must not be deleted until the
remote release assets have been uploaded and verified.
