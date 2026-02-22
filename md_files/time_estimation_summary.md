# Time Estimation for Remaining Experiments

## Current Status

**Completed:** 4/9 experiments (44%)
- All domain-specific experiments completed

**Running:** 4 unique experiments
- phase4_hard_negatives_cosine (parallel)
- phase4_hard_negatives_triplet  
- phase4_bge_large
- phase4_hard_negatives_5neg

## Methodology

Time estimates are based on:
1. **Completed Experiments**: Domain-specific models took 0.5-1.7 hours
2. **Current Runtime**: Running experiments have been active 42-94 hours
3. **Checkpoint Status**: All still in hard negative mining phase
4. **Expected Phases**:
   - Hard Negative Mining: 100-150+ hours
   - Training: ~1.5 hours per epoch

## Key Factors

### Hard Negative Mining Phase
- **Most time-consuming phase** (50-100+ hours)
- Computationally expensive (embedding all corpus passages)
- All 4 experiments currently in this phase
- Progress cannot be resumed if stopped (no checkpoints yet)

### Training Phase
- Relatively fast once mining completes
- Estimated: 1-2 hours per epoch
- Checkpoints saved every 1000 steps
- Can be resumed if paused after first checkpoint

## Estimate Breakdown

See detailed analysis script output for current estimates.

