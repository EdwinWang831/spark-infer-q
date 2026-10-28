# E57: unrotated asymmetric affine4 codec and bounded residual screen

2026-10-28. Preserve E55/E56 negative outcomes, 88GiB process cap, 8GiB host
reserve, same resident worker/objects, serialized queue and exact recovery.
No reload/download/allocator flush/parallel GPU jobs or fixed-draft comparison.

Implement an affine4 codec with actual packed uint8 nibbles and FP16 scale/min.
K groups16/32 successive tokens independently per channel; V groups16/32
channels per token. Compare row-wise K32/V32 as a control. No H32 rotation.
Validate actual encode/decode, packed bytes, constants and axis orientation,
CPU/GPU agreement, Graph replay, and changed inputs against explicit scalar
quantization calculations before full-model screening.

Screen six predeclared combinations against new matched BF16 teacher tokens:
rowK32/V32 R0; tokenK32/V32 R128; tokenK32/V16 R128;
tokenK16/V32 R128; tokenK32/V32 R512; tokenK32/V32 R2048.
Here R is a minimum unquantized recent window. Each attention tile takes a
single cut floor(max(0,tile_start+1-R)/Kgroup)*Kgroup. It quantizes only groups
strictly before the earliest query's residual: no query sees a scale fitted
using future keys. Later queries in a tile keep up to255 extra BF16 rows,
and group alignment adds at most31. Charge R+1024 prepare+256 tile rows in
capacity estimates, retain all codes/metadata, and do not claim a minimal layout.

Use the same E56 four4K/8K development inputs, not a newly held-out set. Store
full <=64-token greedy outputs, BF16 forced logits up to16 positions, original
KL<=.02/L2<=.03/top1>=.95/no-lost-objective decisions. Identity hook must match
BF16 full logits. Counter-check every attention invocation and compression cut.
No heldout/256K quality promotion follows from a development screen.

This full-model screen keeps original BF16 caches and uses packed encode,
dequantize and BF16 work buffers at attention reads. It is not a compressed
resident state or a direct INT4 attention kernel. It changes full-model numerics
causally but cannot establish runtime memory reduction or speed. If a candidate
passes, select minimum forecast storage, then KL, then name; only then proceed
to a directly-reading compressed attention kernel, partial-commit/snapshot/Graph
state tests and fresh long-input validation. If none passes, preserve all results
and stop this candidate family rather than implement an unqualified kernel.

The six reconstruction candidates failed the development gate. A separately
sealed extension now tests whether prompt-time accumulation is material:
compare native BF16, native INT4 throughout, and BF16 prefill followed by native
H32/group32 INT4 KV conversion and INT4 eager decode. Convert in4096-row chunks,
replace each cache once, update dtype/row_bytes/version, preserve all other
state, and fence before any captured-pointer reuse. No old Graph is reused.
Require unchanged non-KV state, direct storage measurement and matched teachers.
Report both all positions and decode positions excluding the uncompressed
prefill logit. Retain four original development prompts and add two new16K
English/Chinese tutorial inputs (up to128 outputs/32 teacher positions). These
new cases are not used to tune the method. This is real compressed decode,
not the prior affine reconstruction screen. It does not prove low peak-memory
prefill admission, Graph/speculation or256K behavior. Keep the original gates.
