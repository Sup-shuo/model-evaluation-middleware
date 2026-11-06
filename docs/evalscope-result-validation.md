EvalScope result validation
===========================

Benchmarks may opt into `protocol.evalscope_result_validation` with a positive integer `subset_counts` mapping and `require_images: true`. MMStar enables all six official subsets at 250 each. Normalization requires matching report counts and unique, matching prediction/review identities, image-bearing inputs, structured outputs and numeric review scores. Existing generation-error/context-overflow rejection remains shared; wrong answers and ordinary max-token stops remain valid.

The binding captures the requested sample limit in task metadata. Integer limits apply per selected subset (all subsets by default), up to the subset size. Fractional limits in (0, 1] use `int(subset_size * fraction)`. Tasks started before this policy keep their original contract. When comparing older runs, check their saved sample artifacts; upgrading does not retroactively validate them. BBH and other benchmarks that do not opt in are unchanged.

Validation streams each artifact once and stores only sample identities/counts; it does not hash artifacts or decode images. Image presence establishes an image-bearing input, not a fresh model-capability probe.
