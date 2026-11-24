# Frozen-model evaluation: tune subset

752 images, 560 lesions, 559 resampling groups, 136 malignant/premalignant (18.09%).
95% percentile intervals from 2000 paired group-bootstrap resamples (seed 42). They condition on the frozen models, calibrators and thresholds.

## Discrimination and operating point

| Model | AUROC | AUPRC | Sensitivity | Specificity | Flagged |
|---|---|---|---|---|---|
| Metadata baseline | 0.803 [0.750, 0.852] | 0.455 [0.355, 0.576] | 0.919 [0.858, 0.967] | 0.550 [0.500, 0.602] | 0.53 |
| Classical ML | 0.845 [0.804, 0.885] | 0.510 [0.408, 0.628] | 0.904 [0.845, 0.957] | 0.594 [0.549, 0.642] | 0.50 |
| CNN | 0.872 [0.835, 0.906] | 0.537 [0.428, 0.656] | 0.904 [0.842, 0.958] | 0.706 [0.660, 0.754] | 0.40 |

Sensitivity and specificity are at each model's frozen threshold.

## Calibration before and after recalibration

| Model | Brier original | Brier recalibrated | Difference (primary) | Verdict | ECE original | ECE recalibrated | Slope / intercept original | Slope / intercept recalibrated |
|---|---|---|---|---|---|---|---|---|
| Metadata baseline | 0.190 [0.172, 0.210] | 0.123 [0.103, 0.143] | -0.0670 [-0.0922, -0.0420] | improvement | 0.244 [0.208, 0.279] | 0.054 [0.033, 0.090] | 1.02 / -1.56 | 1.45 / 0.56 |
| Classical ML | 0.160 [0.141, 0.179] | 0.112 [0.095, 0.130] | -0.0480 [-0.0675, -0.0291] | improvement | 0.175 [0.141, 0.211] | 0.024 [0.019, 0.057] | 0.91 / -1.34 | 1.05 / 0.01 |
| CNN | 0.172 [0.150, 0.194] | 0.107 [0.092, 0.123] | -0.0647 [-0.0862, -0.0443] | improvement | 0.216 [0.181, 0.251] | 0.035 [0.028, 0.067] | 1.19 / -1.78 | 1.22 / -0.07 |

Difference is recalibrated minus original; negative favours recalibration.

## Paired differences between models

| Comparison | AUROC | AUPRC | Sensitivity | Specificity | Brier (recalibrated) |
|---|---|---|---|---|---|
| Classical ML minus Metadata baseline | 0.042 [-0.012, 0.095] | 0.056 [-0.070, 0.171] | -0.015 [-0.080, 0.052] | 0.044 [-0.013, 0.102] | -0.011 [-0.023, 0.001] |
| CNN minus Metadata baseline | 0.069 [0.023, 0.117] | 0.082 [-0.034, 0.191] | -0.015 [-0.085, 0.059] | 0.156 [0.105, 0.210] | -0.016 [-0.029, -0.003] |
| CNN minus Classical ML | 0.027 [-0.010, 0.061] | 0.026 [-0.075, 0.126] | 0.000 [-0.066, 0.060] | 0.112 [0.073, 0.153] | -0.005 [-0.016, 0.006] |

An interval that excludes zero indicates a difference on these cases; one that includes zero does not show that the models are equivalent.

## Flag rate by diagnosis

| Diagnosis | Group | Images | Metadata baseline | Classical ML | CNN |
|---|---|---|---|---|---|
| akiec | malignant/premalignant | 30 | 1.000 [0.886, 1.000] | 1.000 [0.886, 1.000] | 1.000 [0.886, 1.000] |
| bcc | malignant/premalignant | 41 | 0.951 [0.839, 0.987] | 0.951 [0.839, 0.987] | 0.927 [0.806, 0.975] |
| mel | malignant/premalignant | 65 | 0.862 [0.757, 0.925] | 0.831 [0.722, 0.903] | 0.846 [0.739, 0.914] |
| bkl | benign | 86 | 0.837 [0.745, 0.900] | 0.802 [0.706, 0.873] | 0.733 [0.631, 0.815] |
| df | benign | 12 | 0.583 [0.320, 0.807] | 1.000 [0.758, 1.000] | 0.833 [0.552, 0.953] |
| nv | benign | 504 | 0.381 [0.340, 0.424] | 0.327 [0.288, 0.370] | 0.202 [0.170, 0.240] |
| vasc | benign | 14 | 0.429 [0.214, 0.674] | 0.286 [0.117, 0.546] | 0.429 [0.214, 0.674] |

For malignant/premalignant diagnoses the flag rate is sensitivity; for benign ones it is the false-positive rate. Wilson intervals, treating images as independent.
