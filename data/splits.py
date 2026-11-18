"""Frozen FeTA Dataset Splits for Actual 2-Stage Protocol.

Protocol:
1. Development & Epoch Selection:
   - train_56 (56 cases, G1+G2): Model training
   - val_8 (8 cases, G3): Validation to determine best epoch E*
2. Final Training:
   - train_64 (64 cases, G1+G2+G3): Retrain from scratch for E* epochs
3. Evaluation:
   - test_16 (16 cases, G4): Independent holdout test set
"""

SPLIT_ID = 'feta_split_v1_frozen_metadata_balanced'
SEED = 20260918

# Phase 1: 56 train + 8 val for epoch selection
TRAIN_56_CASES = ['sub-001', 'sub-002', 'sub-008', 'sub-009', 'sub-010', 'sub-012', 'sub-013', 'sub-014', 'sub-015', 'sub-016', 'sub-017', 'sub-018', 'sub-020', 'sub-021', 'sub-022', 'sub-023', 'sub-025', 'sub-026', 'sub-028', 'sub-030', 'sub-031', 'sub-032', 'sub-033', 'sub-034', 'sub-035', 'sub-037', 'sub-038', 'sub-039', 'sub-041', 'sub-042', 'sub-043', 'sub-044', 'sub-046', 'sub-047', 'sub-048', 'sub-049', 'sub-050', 'sub-051', 'sub-053', 'sub-054', 'sub-055', 'sub-056', 'sub-058', 'sub-059', 'sub-062', 'sub-063', 'sub-065', 'sub-066', 'sub-067', 'sub-068', 'sub-069', 'sub-070', 'sub-072', 'sub-073', 'sub-077', 'sub-079']
VAL_8_CASES = ['sub-019', 'sub-027', 'sub-029', 'sub-040', 'sub-074', 'sub-075', 'sub-076', 'sub-078']

# Phase 2: 64 combined cases for final retraining
TRAIN_64_CASES = ['sub-001', 'sub-002', 'sub-008', 'sub-009', 'sub-010', 'sub-012', 'sub-013', 'sub-014', 'sub-015', 'sub-016', 'sub-017', 'sub-018', 'sub-019', 'sub-020', 'sub-021', 'sub-022', 'sub-023', 'sub-025', 'sub-026', 'sub-027', 'sub-028', 'sub-029', 'sub-030', 'sub-031', 'sub-032', 'sub-033', 'sub-034', 'sub-035', 'sub-037', 'sub-038', 'sub-039', 'sub-040', 'sub-041', 'sub-042', 'sub-043', 'sub-044', 'sub-046', 'sub-047', 'sub-048', 'sub-049', 'sub-050', 'sub-051', 'sub-053', 'sub-054', 'sub-055', 'sub-056', 'sub-058', 'sub-059', 'sub-062', 'sub-063', 'sub-065', 'sub-066', 'sub-067', 'sub-068', 'sub-069', 'sub-070', 'sub-072', 'sub-073', 'sub-074', 'sub-075', 'sub-076', 'sub-077', 'sub-078', 'sub-079']

# Phase 3: 16 holdout test cases for final evaluation
TEST_16_CASES = ['sub-003', 'sub-004', 'sub-005', 'sub-006', 'sub-007', 'sub-011', 'sub-024', 'sub-036', 'sub-045', 'sub-052', 'sub-057', 'sub-060', 'sub-061', 'sub-064', 'sub-071', 'sub-080']

# Component groups
G1_CASES = ['sub-002', 'sub-008', 'sub-009', 'sub-010', 'sub-012', 'sub-013', 'sub-016', 'sub-017', 'sub-021', 'sub-022', 'sub-023', 'sub-026', 'sub-031', 'sub-033', 'sub-034', 'sub-038', 'sub-041', 'sub-042', 'sub-043', 'sub-046', 'sub-048', 'sub-050', 'sub-051', 'sub-053', 'sub-058', 'sub-062', 'sub-066', 'sub-067', 'sub-068', 'sub-069', 'sub-073', 'sub-077']
G2_CASES = ['sub-001', 'sub-014', 'sub-015', 'sub-018', 'sub-020', 'sub-025', 'sub-028', 'sub-030', 'sub-032', 'sub-035', 'sub-037', 'sub-039', 'sub-044', 'sub-047', 'sub-049', 'sub-054', 'sub-055', 'sub-056', 'sub-059', 'sub-063', 'sub-065', 'sub-070', 'sub-072', 'sub-079']
G3_CASES = ['sub-019', 'sub-027', 'sub-029', 'sub-040', 'sub-074', 'sub-075', 'sub-076', 'sub-078']
G4_CASES = ['sub-003', 'sub-004', 'sub-005', 'sub-006', 'sub-007', 'sub-011', 'sub-024', 'sub-036', 'sub-045', 'sub-052', 'sub-057', 'sub-060', 'sub-061', 'sub-064', 'sub-071', 'sub-080']

# Aliases for backward compatibility
TRAIN_CASES = TRAIN_56_CASES
VAL_CASES = VAL_8_CASES
DEV_CASES = TRAIN_64_CASES
TEST_CASES = TEST_16_CASES
VAL_G2_CASES = G2_CASES
VAL_G3_CASES = G3_CASES
