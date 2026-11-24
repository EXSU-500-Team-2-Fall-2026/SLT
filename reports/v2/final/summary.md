# Frozen-model evaluation: test subset

1511 images, 1222 lesions, 1222 resampling groups, 307 malignant/premalignant (20.32%).
95% percentile intervals from 2000 paired group-bootstrap resamples (seed 42). They condition on the frozen models, calibrators and thresholds.

## Discrimination and operating point

| Model | AUROC | AUPRC | Sensitivity | Specificity | Flagged |
|---|---|---|---|---|---|
| Metadata baseline | 0.726 [0.683, 0.763] | 0.435 [0.368, 0.514] | 0.739 [0.678, 0.797] | 0.674 [0.641, 0.706] | 0.41 |
| Classical ML | 0.819 [0.795, 0.846] | 0.488 [0.426, 0.557] | 0.922 [0.886, 0.954] | 0.536 [0.506, 0.566] | 0.56 |
| CNN | 0.860 [0.835, 0.884] | 0.567 [0.497, 0.638] | 0.860 [0.814, 0.904] | 0.696 [0.665, 0.725] | 0.42 |

Sensitivity and specificity are at each model's frozen threshold.

## Calibration before and after recalibration

| Model | Brier original | Brier recalibrated | Difference (primary) | Verdict | ECE original | ECE recalibrated | Slope / intercept original | Slope / intercept recalibrated |
|---|---|---|---|---|---|---|---|---|
| Metadata baseline | 0.176 [0.163, 0.190] | 0.144 [0.128, 0.160] | -0.0321 [-0.0475, -0.0178] | improvement | 0.186 [0.165, 0.211] | 0.056 [0.040, 0.082] | 0.56 / -1.04 | 0.81 / 0.14 |
| Classical ML | 0.178 [0.166, 0.189] | 0.128 [0.115, 0.140] | -0.0496 [-0.0627, -0.0361] | improvement | 0.187 [0.163, 0.210] | 0.014 [0.014, 0.040] | 0.89 / -1.29 | 1.02 / 0.03 |
| CNN | 0.176 [0.164, 0.188] | 0.117 [0.105, 0.128] | -0.0589 [-0.0722, -0.0455] | improvement | 0.209 [0.187, 0.232] | 0.023 [0.019, 0.048] | 1.19 / -1.60 | 1.22 / 0.12 |

Difference is recalibrated minus original; negative favours recalibration.

## Paired differences between models

| Comparison | AUROC | AUPRC | Sensitivity | Specificity | Brier (recalibrated) |
|---|---|---|---|---|---|
| Classical ML minus Metadata baseline | 0.094 [0.052, 0.144] | 0.053 [-0.022, 0.126] | 0.182 [0.122, 0.252] | -0.138 [-0.180, -0.093] | -0.016 [-0.025, -0.007] |
| CNN minus Metadata baseline | 0.135 [0.094, 0.179] | 0.132 [0.053, 0.197] | 0.121 [0.055, 0.189] | 0.022 [-0.018, 0.064] | -0.027 [-0.037, -0.017] |
| CNN minus Classical ML | 0.041 [0.018, 0.063] | 0.079 [0.015, 0.138] | -0.062 [-0.109, -0.016] | 0.160 [0.128, 0.190] | -0.011 [-0.018, -0.004] |

An interval that excludes zero indicates a difference on these cases; one that includes zero does not show that the models are equivalent.

## Flag rate by diagnosis

| Diagnosis | Group | Images | Metadata baseline | Classical ML | CNN |
|---|---|---|---|---|---|
| akiec | malignant/premalignant | 43 | 0.907 [0.784, 0.963] | 1.000 [0.918, 1.000] | 1.000 [0.918, 1.000] |
| bcc | malignant/premalignant | 93 | 0.624 [0.522, 0.715] | 0.946 [0.880, 0.977] | 0.871 [0.788, 0.925] |
| mel | malignant/premalignant | 171 | 0.760 [0.691, 0.818] | 0.889 [0.833, 0.928] | 0.819 [0.754, 0.869] |
| bkl | benign | 217 | 0.535 [0.468, 0.600] | 0.797 [0.739, 0.845] | 0.710 [0.646, 0.766] |
| df | benign | 44 | 0.114 [0.050, 0.240] | 0.795 [0.655, 0.888] | 0.500 [0.358, 0.642] |
| nv | benign | 908 | 0.293 [0.264, 0.323] | 0.362 [0.332, 0.394] | 0.182 [0.158, 0.208] |
| vasc | benign | 35 | 0.171 [0.081, 0.327] | 0.629 [0.463, 0.768] | 0.714 [0.549, 0.837] |

For malignant/premalignant diagnoses the flag rate is sensitivity; for benign ones it is the false-positive rate. Wilson intervals, treating images as independent.
