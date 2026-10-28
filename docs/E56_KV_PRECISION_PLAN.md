# E56: diagnose and protect sensitive attention layers

2026-10-28. Keep E55's failed quality decisions and unchanged thresholds:
mean KL<=.02, raw relative L2<=.03, top1>=.95, no lost BF16 objectives.
No model reload, download, allocator flush/limit change or parallel GPU work.
Use the existing resident serialized queue and exact baseline recovery audit.

First evaluate all-INT4 versus BF16 using the original six short E47 cases.
Then independently restore BF16 K/V to one attention layer at a time, keeping
all other attention layers INT4. Teacher tokens always come from the BF16
control. Rank layers using calibration teacher-logit L2 reduction only;
do not select layers using held-out cases. Preserve negative contributions.
Protect the top1,2,4 layers as fixed candidates. This is an actual mixed cache
using unchanged production per-layer BF16/INT4 operators, not fake quantization.
State-wide dtype remains the base format for provider compatibility; charge
actual per-cache storage, never its homogeneous cache_bytes estimate.

Evaluate frozen masks on the two E47 4096-token bilingual cases and two new
8192-token English/Chinese ledger + structured-output cases. Require full
<=64 greedy outputs and BF16-forced teacher positions for both controls and
all three candidates. Use freshly matched BF16 teacher tokens (up to16 positions); E55 did not save
short-screen full logits, so do not claim exact old-logit reproduction. Do not
claim 256K, Graph, adaptive or generic quality from this bounded target-only screen.

Measure actual short-state storage and compute the exact 262160-row KV bytes
for each mask. Add the incremental eight-state charge to the E55 INT4 complete
forecast. A candidate that improves precision but worsens an already over-limit
forecast does not solve eight-way capacity. H32/group32 is already in E55;
protecting sensitive layers is a diagnostic and candidate, not an assumed fix.
If it fails, retain the result and use it to choose further K/V asymmetry,
scale-fitting or recent-row residual work. Never describe those as implemented.

After the predeclared layer screen failed, add a separately sealed causal probe:
keep BF16 storage and reconstruct newly prepared K, V or both through the same
H32/group32 INT4 reference quantizer. Include a BF16 control and H32-roundtrip
control. A same-layout scale candidate searches per-group factors
0.85/0.9/0.95/1/1.05/1.1 by local rotated reconstruction MSE, without labels or
heldout-dependent factor selection. Evaluate the four heldout prompts against
new matched BF16 teacher tokens, no greedy quality/default or capacity claim.
This probe isolates numerical effects; its BF16 reconstructed storage and
unrotated query/output arithmetic are not a deployable INT4 kernel. Record
actual INT4 and simulated-both separately; do not equate their metrics.
