# GRAFT-FL Time-4 complete-test preparation: verified source mismatch and recovery

**Date:** 2026-11-13, America/New_York  
**Research branch:** `graft-fl-aiscalar-20261113`  
**Status:** Source/root-cause verified from original upstream and official Defects4J metadata. Fixes and regression tests committed. **No new AIScalar execution or Qwen inference yet.**

## Observed failure

`bash scripts/aiscalar_graft_phase3_time4_full_test.sh` passed 32 tests but stopped before inference:

```text
ValueError: No complete parsed method body for org.joda.time.TestPartial_Basics::testWith3
```

The extraction code was asked to read `testWith3()` from the frozen Time-4 **raw buggy Git checkout**. This method is genuinely absent from that revision. It is not the same as `testWith3a()` or `testWith3b()`, which are present.

## Independent source verification

Official Defects4J metadata:

- Bug ID: `Time-4`
- Raw buggy production revision: `bcb044669b4d1f8d334861ccbd169924d6ef3b54`
- Fixed revision: `3ba9ba799b3261b7332a467a88be142c83b298fd`
- Official test patch: [Time/patches/4.test.patch](https://github.com/rjust/defects4j/blob/master/framework/projects/Time/patches/4.test.patch)
- Official observed trigger metadata: [Time/trigger_tests/4](https://github.com/rjust/defects4j/blob/master/framework/projects/Time/trigger_tests/4)
- Official revision manifest: [Time/active-bugs.csv](https://github.com/rjust/defects4j/blob/master/framework/projects/Time/active-bugs.csv)

I inspected both immutable Joda-Time sources:

1. [Raw buggy test file at bcb04466](https://github.com/JodaOrg/joda-time/blob/bcb044669b4d1f8d334861ccbd169924d6ef3b54/src/test/java/org/joda/time/TestPartial_Basics.java): `testWith3()` **absent**. Nearby tests `testWith2()`, `testWith3a()` etc. exist.
2. [Fixed-revision test file at 3ba9ba79](https://github.com/JodaOrg/joda-time/blob/3ba9ba799b3261b7332a467a88be142c83b298fd/src/test/java/org/joda/time/TestPartial_Basics.java): complete `testWith3()` **present**, including `catch (IllegalArgumentException ...)`.

This is a legitimate **benchmark triggering-test provenance distinction**: the test is recorded by Defects4J but is not present in the original raw buggy repository commit. It is incorrect to assume the buggy checkout itself contains this test.

## Correction, controlled and gold-free

`scripts/prepare_graft_time4_authentic_test.py` now:

1. Enforces exactly the frozen Time-4 buggy revision and exactly the official test ID.
2. Checks the raw buggy test file to establish that `testWith3()` is absent.
3. Reads only the `src/test/java/org/joda/time/TestPartial_Basics.java` **test file** from pinned fixed Git commit `3ba9ba...` using local `git show`. It reads **no fixed production source**, and never switches the checked-out buggy revision.
4. AST-extracts exactly `testWith3()` and verifies the original trigger's named API call, `fail()`, and `IllegalArgumentException` catch.
5. Records test source file and method SHA-256 hashes, exact revision, original trigger SHA-256, official patch reference, no new tests run, and no checkout modification.
6. Maintains `evidence_kind=trigger_description`, because the script does not execute a new failing test. The original trigger text remains unchanged and is labelled as original supplied evidence.
7. Fails without Qwen calls if the pinned fixed test Git object is unavailable locally. It does **not** silently download a different release or use `testWith3a()`.

`scripts/prepare_graft_phase3_real_bugs.py::_one_test_method` now validates the **target AST method body**, not requiring error-free parsing of unrelated source elsewhere. It still rejects syntactically invalid or partial target methods.

Regression coverage:
- `tests/test_graft_time4_official_trigger.py`: isolated local Git repo showing a trigger added only in fixed test revision, no change to buggy checkout, no use of fixed production code, and fail-closed wrong revisions.
- `tests/test_graft_phase3_real_bugs.py`: exact method identity and rejection of incomplete method code.

## Next on AIScalar

```bash
cd ~/RAVE_FL_Research/GRAFT_FL_New_Run
conda activate rave-fl
git fetch origin graft-fl-aiscalar-20261113
git merge --ff-only FETCH_HEAD

python -m pytest -q \
  tests/test_graft_time4_official_trigger.py \
  tests/test_graft_phase3_real_bugs.py

bash scripts/aiscalar_graft_phase3_time4_full_test.sh
```

The full runner rechecks regression tests, prepares the test-only evidence, checks source and provenance, then preflights and runs Qwen if all gates pass. **It has not been rerun yet.**

If it reports that fixed commit `3ba9ba...` is not in the local Git object store, verify it in the immutable Time-4 repository using `git cat-file -e`. Obtain the exact pinned revision only via an authorized project/benchmark mirror, without switching or modifying the buggy checkout. The recovery must retain source SHA/revision hashes.

## Scientific interpretation

Do not call the new complete test `buggy-revision test source`. Its provenance is `official Defects4J triggering test read from a fixed-revision TEST file`. Only the target program code under localization stays at the raw buggy revision. This follows the benchmark's test provenance and does not rely on model outputs or gold method labels.

This changes the previously blocked experimental input condition, so comparisons with the original truncated `trigger_description` must be labelled clearly as `full official trigger test vs abbreviated trigger`. Time-4 remains repeatedly studied **development data**, not independent held-out validation.
