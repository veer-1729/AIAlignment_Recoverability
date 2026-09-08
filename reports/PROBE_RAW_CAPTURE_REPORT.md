PARTIAL CAPTURE -- split scheme 1 of 2 only (grouped_by_rollout).
Relayed from the EC2 run at etimes 15717 while scheme 2 (held_out_game) was still running.
NOT AUTHORITATIVE. Superseded by the complete stdout + reports/probe_study_scale.json
with sha256sums when run 1 exits. Kept only so the numbers survive a session loss.

checkpoints: 1971   (activations x derived values)

[VARIANCE]  mean / sd / range
  Q_continue      -0.506   1.014  [-1.49, 0.99]
  Q_intervene      0.330   0.895  [-1.54, 0.94]
  tau              0.836   1.039  [-2.06, 2.41]

##########################################################################
# SPLIT SCHEME: grouped_by_rollout   (504 groups)
##########################################################################

--- seed 0 (detailed, with paired bootstrap CIs) ---
  train=1381  test=590  (353 train groups, 151 test groups)

  TARGET Q_continue [regression]
    observables          R2  0.497  [ 0.450,  0.540]
    act_L8               R2  0.622  [ 0.577,  0.662]   D +0.124 [+0.090, +0.160] *
    obs+act_L8           R2  0.627  [ 0.582,  0.667]   D +0.129 [+0.096, +0.164] *
    act_L16              R2  0.668  [ 0.625,  0.706]   D +0.170 [+0.129, +0.211] *
    obs+act_L16          R2  0.670  [ 0.628,  0.709]   D +0.173 [+0.132, +0.212] *
    act_L24              R2  0.731  [ 0.687,  0.772]   D +0.234 [+0.190, +0.277] *
    obs+act_L24          R2  0.732  [ 0.688,  0.773]   D +0.235 [+0.191, +0.277] *

  TARGET Q_intervene [regression]
    observables          R2  0.515  [ 0.454,  0.570]
    act_L8               R2  0.583  [ 0.517,  0.642]   D +0.068 [+0.025, +0.114] *
    obs+act_L8           R2  0.582  [ 0.519,  0.640]   D +0.068 [+0.029, +0.110] *
    act_L16              R2  0.625  [ 0.560,  0.680]   D +0.110 [+0.050, +0.170] *
    obs+act_L16          R2  0.627  [ 0.564,  0.681]   D +0.112 [+0.054, +0.167] *
    act_L24              R2  0.701  [ 0.656,  0.742]   D +0.186 [+0.134, +0.240] *
    obs+act_L24          R2  0.706  [ 0.661,  0.747]   D +0.192 [+0.139, +0.245] *

  TARGET tau [regression]
    observables          R2  0.318  [ 0.260,  0.376]
    act_L8               R2  0.467  [ 0.414,  0.520]   D +0.149 [+0.096, +0.206] *
    obs+act_L8           R2  0.466  [ 0.413,  0.520]   D +0.148 [+0.096, +0.204] *
    act_L16              R2  0.518  [ 0.467,  0.569]   D +0.201 [+0.148, +0.254] *
    obs+act_L16          R2  0.517  [ 0.464,  0.567]   D +0.199 [+0.146, +0.252] *
    act_L24              R2  0.557  [ 0.502,  0.607]   D +0.239 [+0.186, +0.292] *
    obs+act_L24          R2  0.561  [ 0.506,  0.611]   D +0.243 [+0.191, +0.296] *

  TARGET continue_success [classification]
    observables          acc  0.898  macroF1  0.890  [ 0.863,  0.916]
    act_L8               acc  0.892  macroF1  0.882  [ 0.854,  0.908]   D -0.008 [-0.035, +0.019]
    obs+act_L8           acc  0.895  macroF1  0.886  [ 0.857,  0.911]   D -0.005 [-0.036, +0.027]
    act_L16              acc  0.903  macroF1  0.895  [ 0.868,  0.919]   D +0.004 [-0.027, +0.033]
    obs+act_L16          acc  0.903  macroF1  0.895  [ 0.868,  0.919]   D +0.005 [-0.026, +0.034]
    act_L24              acc  0.920  macroF1  0.913  [ 0.890,  0.936]   D +0.023 [-0.007, +0.053]
    obs+act_L24          acc  0.924  macroF1  0.917  [ 0.894,  0.940]   D +0.027 [-0.003, +0.057]

  TARGET oracle_action [classification]
    observables          acc  0.761  macroF1  0.762  [ 0.727,  0.795]
    act_L8               acc  0.795  macroF1  0.797  [ 0.765,  0.829]   D +0.035 [-0.002, +0.074]
    obs+act_L8           acc  0.797  macroF1  0.799  [ 0.767,  0.831]   D +0.037 [+0.002, +0.074] *
    act_L16              acc  0.808  macroF1  0.813  [ 0.782,  0.842]   D +0.051 [+0.013, +0.090] *
    obs+act_L16          acc  0.815  macroF1  0.819  [ 0.788,  0.848]   D +0.057 [+0.021, +0.095] *
    act_L24              acc  0.839  macroF1  0.840  [ 0.809,  0.869]   D +0.078 [+0.039, +0.115] *
    obs+act_L24          acc  0.842  macroF1  0.843  [ 0.813,  0.872]   D +0.081 [+0.044, +0.117] *

  ----------------------------------------------------------------------
  CONDITIONAL TESTS -- tau given continuation prospects  [Q2]
  ----------------------------------------------------------------------
    [C1] baseline = observables + Q_C          R2  0.681  (alpha 10)
    [C1]   + activations L8                   R2  0.712   D +0.031 [-0.005, +0.068]    (alpha 100)
    [C1]   + activations L16                  R2  0.731   D +0.050 [+0.011, +0.092] *  (alpha 100)
    [C1]   + activations L24                  R2  0.726   D +0.045 [+0.002, +0.089] *  (alpha 100)
    [C2] residual tau after observables + Q_C  (resid var 0.3419 of total 1.0954)
    [C2]   activations L8  -> residual       R2  0.123  [ 0.025,  0.208] *
    [C2]   activations L16 -> residual       R2  0.208  [ 0.109,  0.290] *
    [C2]   activations L24 -> residual       R2  0.323  [ 0.230,  0.405] *
    [C3] within continuation-risk bands (n>=40 only)
      risk=0.00  n=598  (te 192)  observables R2  0.297
                    + activations L8        R2  0.134   D -0.163 [-0.223, -0.080] *
                    + activations L16       R2  0.502   D +0.205 [-0.127, +0.285]
                    + activations L24       R2  0.368   D +0.071 [-0.173, +0.166]
      risk=0.80  n=41   (te 13 )  observables R2  0.048
                    + activations L8        R2 -0.006   D -0.054 [-10.789, -0.037] *
                    + activations L16       R2  0.003   D -0.045 [-10.928, -0.026] *
                    + activations L24       R2  0.000   D -0.048 [-18.159, -0.008] *
      risk=1.00  n=1220 (te 348)  observables R2  0.486
                    + activations L8        R2  0.556   D +0.070 [+0.017, +0.123] *
                    + activations L16       R2  0.622   D +0.136 [+0.069, +0.204] *
                    + activations L24       R2  0.689   D +0.203 [+0.142, +0.267] *

--- seeds 1..4 (silent; feeding the stability table) ---

  ======================================================================
  SEED STABILITY over 5 splits -- delta vs observables  [Q4]
  target                   features             mean       sd      sign
  ----------------------------------------------------------------------
  Q_continue               act_L8             +0.079    0.045      same
  Q_continue               obs+act_L8         +0.082    0.048      same
  Q_continue               act_L16            +0.174    0.027      same
  Q_continue               obs+act_L16        +0.176    0.026      same
  Q_continue               act_L24            +0.222    0.030      same
  Q_continue               obs+act_L24        +0.222    0.030      same
  Q_intervene              act_L8             +0.084    0.024      same
  Q_intervene              obs+act_L8         +0.081    0.022      same
  Q_intervene              act_L16            +0.116    0.036      same
  Q_intervene              obs+act_L16        +0.114    0.032      same
  Q_intervene              act_L24            +0.146    0.056      same
  Q_intervene              obs+act_L24        +0.152    0.055      same
  tau                      act_L8             +0.125    0.031      same
  tau                      obs+act_L8         +0.125    0.032      same
  tau                      act_L16            +0.202    0.026      same
  tau                      obs+act_L16        +0.196    0.027      same
  tau                      act_L24            +0.223    0.049      same
  tau                      obs+act_L24        +0.225    0.050      same
  continue_success         act_L8             -0.000    0.019     FLIPS
  continue_success         obs+act_L8         +0.003    0.022     FLIPS
  continue_success         act_L16            +0.022    0.016      same
  continue_success         obs+act_L16        +0.023    0.015      same
  continue_success         act_L24            +0.038    0.013      same
  continue_success         obs+act_L24        +0.037    0.011      same
  oracle_action            act_L8             +0.026    0.014      same
  oracle_action            obs+act_L8         +0.028    0.014      same
  oracle_action            act_L16            +0.059    0.009      same
  oracle_action            obs+act_L16        +0.059    0.008      same
  oracle_action            act_L24            +0.067    0.011      same
  oracle_action            obs+act_L24        +0.064    0.014      same
  ----------------------------------------------------------------------

================================================================================
SPLIT SCHEME 2 (held_out_game) -- decisive extracts relayed at run 1 exit
Run 1 exited cleanly 07:56:12Z after 8h46m.
  sha256 probe_pooled.log      fc6e7bddb40d89b1bcc49c8a718efefd1ff065faab31fc0b2b9a6dc1c66026f7
  sha256 probe_study_scale.json fab11993c36933d22b6771878707ac95797accf07b5ecc9d5ce8693bf03ec154
Full seed-0 text still to be relayed. C3 per-seed stability still to be extracted
from the JSON (it IS present there; seed_summary simply never prints it).
================================================================================

--- SCHEME 1 conditional stability (the six previously-missing rows) ---
  tau | conditioned        C1 +act L8         +0.019    0.030     FLIPS
  tau | conditioned        C1 +act L16        +0.033    0.033     FLIPS
  tau | conditioned        C1 +act L24        +0.015    0.056     FLIPS
  tau | conditioned        C2 resid L8        +0.138    0.044      same
  tau | conditioned        C2 resid L16       +0.192    0.050      same
  tau | conditioned        C2 resid L24       +0.276    0.069      same

--- SCHEME 2 unconditional targets, L24, 5-seed stability ---
  Q_continue     +0.174  sd 0.034  same    (scheme 1: +0.222)
  Q_intervene    +0.120  sd 0.036  same    (scheme 1: +0.146)
  tau            +0.160  sd 0.045  same    (scheme 1: +0.223)
  oracle_action  +0.046  sd 0.026  same    (scheme 1: +0.067)   [L8, L16 FLIP]
  continue_success: ALL SIX ROWS FLIP -- complete null under scheme 2

--- SCHEME 2 conditional tests, seed 0 ---
  [C1] baseline observables + Q_C            R2  0.724
  [C1]   + activations L8                    R2  0.668   D -0.056 [-0.098, -0.012] *
  [C1]   + activations L16                   R2  0.646   D -0.078 [-0.125, -0.027] *
  [C1]   + activations L24                   R2  0.657   D -0.067 [-0.111, -0.020] *
  [C2] residual tau after observables + Q_C  (resid var 0.2845 of total 1.0314)
  [C2]   activations L8  -> residual        R2  0.057  [ 0.023,  0.086] *
  [C2]   activations L16 -> residual        R2  0.114  [-0.006,  0.213]
  [C2]   activations L24 -> residual        R2  0.279  [ 0.172,  0.374] *

--- SCHEME 2 conditional stability ---
  tau | conditioned        C1 +act L8         -0.057    0.029      same   (consistently NEGATIVE)
  tau | conditioned        C1 +act L16        -0.040    0.051     FLIPS
  tau | conditioned        C1 +act L24        -0.077    0.063     FLIPS
  tau | conditioned        C2 resid L8        +0.018    0.027     FLIPS
  tau | conditioned        C2 resid L16       +0.039    0.062     FLIPS
  tau | conditioned        C2 resid L24       +0.197    0.118     FLIPS

--- THE ARBITER: C3 risk=1.00 band, seed 0 ---
  scheme 1:  L8 +0.070 *              L16 +0.136 *              L24 +0.203 *
  scheme 2:  L8 +0.030 [-0.033,+0.088]  L16 +0.049 [-0.019,+0.114]  L24 +0.079 [+0.015,+0.146] *
  Survives at L24 only, and collapses 61% from +0.203 to +0.079.
  SEED STABILITY NOT YET COMPUTED -- pending JSON extraction.

--- INTERPRETIVE NOTE, recorded before the writeup ---
C1's negativity under scheme 2 is NOT a finding. The pre-registration states that
C1 fits one ridge alpha over 27 observable columns plus 3,584 activation dims, that
the activation block drags alpha up and shrinks the standardised Q_C coefficient,
and that "C1 can lose for a reason that has nothing to do with activation content."
C1 behaving badly here is the predicted artifact. C2 is the arbiter for Q2, and
C2 FLIPS at all three layers under scheme 2 -- a null.

================================================================================
C3 PER-SEED STABILITY, extracted from probe_study_scale.json
(computed every seed by run_pass; seed_summary simply never prints the C3 keys)
================================================================================
grouped_by_rollout
  C3_band_0.00  5/5 seeds  n_test [192,154,191,213,161]  obs R2 [0.297,-0.062,0.17,0.206,0.185]
    L8  -0.151 sd 0.091 FLIPS  [-0.163, 0.007, -0.194, -0.219, -0.187]
    L16 -0.045 sd 0.149 FLIPS  [ 0.205,-0.026, -0.137, -0.167, -0.097]
    L24 -0.092 sd 0.133 FLIPS  [ 0.071, 0.032, -0.196, -0.209, -0.157]
  C3_band_0.20  4/5 seeds  n_test [18,15,11,12]  obs R2 [-0.066,-4.26,-0.14,-3.347]
    L8  +0.413 sd 1.795 FLIPS  [0.0, 2.919, -1.351, 0.085]
    L16 +0.781 sd 1.675 FLIPS  [0.0, 3.29, -0.181, 0.015]
    L24 +0.687 sd 1.594 FLIPS  [-0.008, 3.074, -0.226, -0.091]
  C3_band_0.80  5/5 seeds  n_test [13,11,15,10,11]  obs R2 [0.048,-0.025,0.247,0.17,0.017]
    L8  -0.086 sd 0.107 FLIPS  [-0.054, 0.025, -0.264, -0.082, -0.054]
    L16 -0.094 sd 0.111 FLIPS  [-0.045, 0.031, -0.261, -0.139, -0.053]
    L24 -0.128 sd 0.137 FLIPS  [-0.048, 0.001, -0.274, -0.28, -0.04]
  C3_band_1.00  5/5 seeds  n_test [348,380,348,326,384]  obs R2 [0.486,0.522,0.447,0.512,0.422]
    L8  +0.087 sd 0.020 same   [0.07, 0.065, 0.085, 0.106, 0.11]
    L16 +0.117 sd 0.035 same   [0.136, 0.066, 0.099, 0.123, 0.159]
    L24 +0.158 sd 0.045 same   [0.203, 0.112, 0.135, 0.131, 0.212]

held_out_game
  C3_band_0.00  5/5 seeds  n_test [220,160,156,214,149]  obs R2 [0.193,-0.05,-2.606,-0.003,0.132]
    L8  +0.375 sd 1.046 FLIPS  [-0.218, -0.011, 2.24, -0.011, -0.124]
    L16 +0.117 sd 0.363 FLIPS  [-0.087, 0.016, 0.762, -0.034, -0.073]
    L24 +0.084 sd 0.313 FLIPS  [-0.11, 0.046, 0.63, -0.019, -0.127]
  C3_band_0.20  5/5 seeds  n_test [15,17,17,11,20]  obs R2 [-0.023,-0.035,-0.059,-0.103,-2.24]
    L8  +0.114 sd 0.161 FLIPS  [0.033, -0.016, 0.019, 0.157, 0.376]
    L16 +0.012 sd 0.058 FLIPS  [-0.033, -0.017, 0.018, 0.111, -0.017]
    L24 -0.162 sd 0.357 FLIPS  [-0.057, -0.033, 0.018, 0.058, -0.796]
  C3_band_0.80  4/5 seeds  n_test [12,13,16,11]  obs R2 [-0.134,-0.004,0.25,-4.419]
    L8  -0.012 sd 0.106 FLIPS  [0.023, 0.021, -0.167, 0.076]
    L16 +0.094 sd 0.236 FLIPS  [0.029, 0.004, -0.096, 0.438]
    L24 -0.302 sd 0.359 FLIPS  [-0.003, 0.002, -0.496, -0.71]
  C3_band_1.00  5/5 seeds  n_test [324,387,384,334,365]  obs R2 [0.577,0.453,0.44,0.404,0.411]
    L8  +0.058 sd 0.018 same   [0.03, 0.065, 0.06, 0.079, 0.058]
    L16 +0.095 sd 0.051 same   [0.049, 0.108, 0.058, 0.175, 0.087]
    L24 +0.122 sd 0.061 same   [0.079, 0.095, 0.136, 0.222, 0.078]

THE ARBITER: held_out_game / C3_band_1.00 is present in 5/5 seeds and sign-consistent
at all three layers -- all fifteen per-seed deltas positive, monotone in depth, on a
healthy cell (n_test 324-387, observables R2 0.404-0.577).
Mean-to-mean reduction vs scheme 1 at L24: +0.158 -> +0.122, about 23%. The earlier
"61% collapse" was a seed-0-to-seed-0 artifact (seed 0 is 2nd-highest of five in
scheme 1 and 2nd-lowest of five in scheme 2) and is withdrawn.
It is the ONLY C3 band that is stable in either scheme.

================================================================================
SCHEME 2 (held_out_game) -- COMPLETE VERBATIM BLOCK THROUGH EOF
train=1383  test=588  (176 train groups, 76 test groups)
================================================================================
  TARGET Q_continue [regression]
    observables          R2  0.356  [ 0.297,  0.412]
    act_L24              R2  0.557  [ 0.503,  0.607]   D +0.201 [+0.152, +0.252] *
    obs+act_L24          R2  0.558  [ 0.504,  0.608]   D +0.202 [+0.154, +0.252] *
  TARGET Q_intervene [regression]
    observables          R2  0.595  [ 0.539,  0.648]
    act_L24              R2  0.699  [ 0.654,  0.741]   D +0.104 [+0.049, +0.160] *
    obs+act_L24          R2  0.706  [ 0.660,  0.747]   D +0.111 [+0.057, +0.166] *
  TARGET tau [regression]
    observables          R2  0.167  [ 0.095,  0.232]
    act_L24              R2  0.363  [ 0.296,  0.420]   D +0.196 [+0.142, +0.253] *
    obs+act_L24          R2  0.365  [ 0.298,  0.422]   D +0.198 [+0.145, +0.254] *
  TARGET continue_success [classification]
    observables          acc  0.845  macroF1  0.841
    obs+act_L24          acc  0.884  macroF1  0.881   D +0.039 [+0.005, +0.073] *   (but FLIPS across seeds)
  TARGET oracle_action [classification]
    observables          acc  0.726  macroF1  0.737
    obs+act_L24          acc  0.789  macroF1  0.796   D +0.059 [+0.021, +0.098] *

SCHEME 2 seed-stability, all 36 rows: see the peer transcript; key rows recorded above.
Full seed-0 detail for every layer/block and every CI is in probe_pooled.log
(sha256 fc6e7bddb40d89b1bcc49c8a718efefd1ff065faab31fc0b2b9a6dc1c66026f7) and every
seed's numbers are in probe_study_scale.json
(sha256 fab11993c36933d22b6771878707ac95797accf07b5ecc9d5ce8693bf03ec154).

================================================================================
PER-SEED VALUES FOR ALL 36 STABILITY ROWS (from probe_study_scale.json)
================================================================================
grouped_by_rollout
  C1_L8   +0.019 sd 0.030 FLIPS  [0.031, -0.007, -0.009, 0.014, 0.064]
  C1_L16  +0.033 sd 0.033 FLIPS  [0.05, -0.007, 0.015, 0.028, 0.079]
  C1_L24  +0.015 sd 0.056 FLIPS  [0.045, -0.04, -0.019, -0.007, 0.098]
  C2_L8   +0.138 sd 0.044 same   [0.123, 0.114, 0.088, 0.193, 0.174]
  C2_L16  +0.192 sd 0.050 same   [0.208, 0.176, 0.113, 0.218, 0.244]
  C2_L24  +0.276 sd 0.069 same   [0.323, 0.232, 0.194, 0.266, 0.365]
  Q_continue|act_L24     +0.222 same [0.234, 0.241, 0.169, 0.24, 0.225]
  Q_intervene|act_L24    +0.146 same [0.186, 0.1, 0.125, 0.095, 0.224]
  tau|act_L24            +0.223 same [0.239, 0.223, 0.143, 0.277, 0.232]
  continue_success|act_L24 +0.038 same [0.023, 0.05, 0.035, 0.052, 0.029]
  oracle_action|act_L24  +0.067 same [0.078, 0.066, 0.05, 0.072, 0.071]
held_out_game
  C1_L8   -0.057 sd 0.029 same   [-0.056, -0.083, -0.018, -0.042, -0.088]   <- consistently NEGATIVE
  C1_L16  -0.040 sd 0.051 FLIPS  [-0.078, -0.027, -0.079, 0.044, -0.059]
  C1_L24  -0.077 sd 0.063 FLIPS  [-0.067, -0.095, -0.057, 0.005, -0.17]
  C2_L8   +0.018 sd 0.027 FLIPS  [0.057, 0.025, 0.013, 0.013, -0.017]
  C2_L16  +0.039 sd 0.062 FLIPS  [0.114, 0.09, 0.01, 0.017, -0.036]
  C2_L24  +0.197 sd 0.118 FLIPS  [0.279, 0.236, 0.238, 0.242, -0.012]
  Q_continue|act_L24     +0.174 same  [0.201, 0.176, 0.134, 0.145, 0.213]
  Q_intervene|act_L24    +0.120 same  [0.104, 0.103, 0.141, 0.172, 0.081]
  tau|act_L24            +0.160 same  [0.196, 0.141, 0.093, 0.204, 0.168]
  continue_success|act_L24 +0.013 FLIPS [0.037, 0.004, 0.004, -0.012, 0.034]
  oracle_action|act_L24  +0.046 same  [0.056, 0.025, 0.024, 0.085, 0.038]
  oracle_action|act_L16  +0.032 FLIPS [0.037, 0.022, -0.002, 0.068, 0.036]
  oracle_action|act_L8   -0.001 FLIPS [0.007, -0.013, -0.011, 0.02, -0.008]

SEED 4 ANOMALY (post-hoc, not part of any verdict): seed index 4 is the sole negative
on all three C2 layers under held_out_game and the most negative on C1_L8 and C1_L24,
while being the HIGHEST of five on Q_continue (0.213), mid-range on tau (0.168 vs
0.093-0.204), and positive on the arbiter band (C3 1.00 L24 = 0.078). C1 and C2 are
the only two tests that condition on Q_C. Pending extraction of per-seed
C1_baseline_r2 / C2_resid_var to test whether that split simply had little residual
variance left to explain.

================================================================================
PER-SEED SPLIT DIAGNOSTICS -- falsifies the residual-variance explanation of seed 4
================================================================================
grouped_by_rollout
  seed ntrain ntest trGrp teGrp C1_base_R2 C1_alpha residVar totVar
     0   1381   590   353   151      0.681       10   0.3419  1.0954
     1   1382   589   353   151      0.683       10   0.3338  1.0717
     2   1385   586   353   151      0.644       10   0.3893  1.0940
     3   1390   581   353   151      0.709       10   0.3270  1.1261
     4   1378   593   353   151      0.575       10   0.4589  1.0812
held_out_game
     0   1383   588   176    76      0.724       10   0.2845  1.0314
     1   1372   599   176    76      0.633       10   0.3930  1.0702
     2   1373   598   176    76      0.574       10   0.4419  1.0377
     3   1379   592   176    76      0.603       10   0.4019  1.0121
     4   1381   590   176    76      0.650       10   0.3332  1.0567

FALSIFIED: the hypothesis was that seed 4's baseline fit unusually well, leaving too
little residual variance for C2 to explain. Under held_out_game seed 0 has the
SMALLEST residual variance (0.2845) and the HIGHEST baseline R2 (0.724) and returns
the BEST C2_L24 (+0.279); seed 4 is 2nd on both diagnostics and returns the only
negative. The predicted direction is reversed at the relevant end. Split sizes are
uniform (176/76 groups every seed) and C1_alpha is 10 throughout, so neither explains
it either. No mechanism identified; none offered.

================================================================================
RUN 2 -- n=928, `original` cohort only. Exit 0 at 2026-09-07T14:19:19Z, 6h23m.
  sha256 probe_original.log            61c5eb82f932be40eeb592021cea9da4d54449c9d3a968f61ceb1778a56504af
  sha256 probe_study_scale_original.json 095d8feb8ae72da0ec46f03c859b4db6547f6a9ddccf6f2a0403d072ee234285
  scheme 1: 417 groups | scheme 2: 250 groups
  [VARIANCE] Q_continue -0.513 sd 1.014 | Q_intervene 0.266 sd 0.921 | tau 0.779 sd 1.027
================================================================================
ARBITER  C3 band 1.00, held_out_game, 5/5 seeds, n_test 155-196, obs R2 0.161-0.485
  L8  +0.070 sd 0.056 same  [0.025, 0.012, 0.147, 0.062, 0.103]
  L16 +0.123 sd 0.042 same  [0.097, 0.115, 0.176, 0.155, 0.073]
  L24 +0.162 sd 0.097 same  [0.055, 0.12, 0.316, 0.174, 0.146]
  (scheme 1 equivalent: L8 +0.091 FLIPS, L16 +0.152 same, L24 +0.162 same)

UNCONDITIONAL, held_out_game        RUN2 (n=928)      RUN1 (n=1971)
  tau           act_L24             +0.158 same       +0.160 same
  tau           act_L16             +0.139 same       +0.152 same
  tau           act_L8              +0.093 same       +0.067 same
  Q_continue    act_L24             +0.122 same       +0.174 same
  Q_intervene   act_L24             +0.103 same       +0.120 same
  oracle_action act_L24             +0.051 same       +0.046 same
  oracle_action act_L16             +0.025 FLIPS      +0.032 FLIPS
  oracle_action act_L8              -0.015 same(neg)  -0.001 FLIPS
  continue_success act_L8           -0.035 same(neg); L16/L24 FLIP -- null again

CONDITIONALS, held_out_game, RUN 2
  C2 resid L8   -0.009 sd 0.020 FLIPS
  C2 resid L16  -0.009 sd 0.021 FLIPS
  C2 resid L24  +0.059 sd 0.086 FLIPS
  C1 +act L8    -0.147 sd 0.054 same
  C1 +act L16   -0.157 sd 0.100 same
  C1 +act L24   -0.163 sd 0.069 same
C2 null replicates independently -> the seed-4 anomaly is NOT load-bearing.
C1 artifact is LARGER at smaller n, as the pre-registered ridge-alpha mechanism predicts.

================================================================================
LEAK-CEILING DIAGNOSTIC (post-hoc, outside the pre-registered study)
  sha256 game_id_ceiling.log        6d8c6919831e28853c3c0bcceaf99f7cb73cf879a9f944a6a85c2cb1c049dd96
  sha256 game_id_ceiling_scale.json 400285fd8b574c0b38cfaace2025f45c7e89fb59beb049cbc8752dc01b2a2dc1
================================================================================
GATE 1  negative control, held_out_game, game_onehot -- PASSES
  Q_continue  R2 -0.020 sd 0.014 (seed0 -0.041)   <= 0  OK
  Q_intervene R2 -0.019 sd 0.027 (seed0 -0.012)   <= 0  OK
  tau         R2 -0.024 sd 0.034 (seed0 -0.014)   <= 0  OK
  seed 0 train/test game overlap: 0.0%
GATE 2  wiring check -- PASSES
  obs+act_L24 seed0  0.732 / 0.706 / 0.561  ==  probe_study reference  0.732 / 0.706 / 0.561

THE CHANNEL, MEASURED
  grouped_by_rollout seed 0 train/test game overlap: 65.6%
  game_onehot alone :  Q_C 0.407 | Q_I 0.317 | tau 0.320
  observables alone :  Q_C 0.441 | Q_I 0.508 | tau 0.246
  -> on tau, PURE GAME IDENTITY BEATS THE 26 OBSERVABLES (0.320 vs 0.246)

THE SHARP ONE -- scheme 1, does the channel explain the advantage?
  target        over obs   over obs+game   retained
  Q_continue     +0.222      +0.194          87%
  Q_intervene    +0.152      +0.138          91%
  tau            +0.225      +0.168          75%
  game identity's own gain over observables: +0.063 / +0.029 / +0.105

SCHEME 2 BLOCK -- NOT COMPARABLE, EXCLUDED FROM THE ARGUMENT
  over obs+game: +0.251 / +0.197 / +0.219, but the one-hot is pure noise there
  (0.0% overlap) and degrades the baseline: game over obs = -0.085 / -0.073 / -0.059.
  Larger figures are a weakened baseline, not a stronger effect.

LIMIT OF THE BOUND: a one-hot encodes per-game MEANS, so it ceilings the ADDITIVE
game-identity channel only. Interactions between game identity and state (game x step)
are not bounded by it and this dataset cannot support them.

================================================================================
CONFLICT-SET ANALYSIS -- is the signal intervention-specific, or a Q_C proxy?
  sha256 conflict_all.log            25b16cc17c5bfc5962bec216381912f20cf192c59edb19031b4f1713f1b29f5f
  sha256 conflict_original.log       2b52159724813a40ccd358948f285470a7841849153eaceb3503bcac7e33b5b5
  sha256 conflict_set_scale_all.json 0a1e5a95e4e737f38fcc03136d496ba6c4355717734eb4ac689c58b5b2d61d6d
  sha256 conflict_set_scale_original.json 25e29ae48e06c327b631f364431488a2b16167c75727f473da7e58fdc073d758
  runtimes: all 34m, original 22m. Both exit 0.
================================================================================
GATES
  cohort=original prints 928 checkpoints  -> cohort filter matches run 2 exactly
  sufficiency OK on all four scheme/cohort blocks, no THIN markers
  oracle continue excluded: 0 in both cohorts

  cohort=all       conflict 1220  (686 intervene / 534 quit)
                   Q_I mean  0.009 sd 0.993 [-1.54, 0.94]
                   Q_C mean -1.250 sd 0.141   <- pinned; ~2% of Q_I's variance
  cohort=original  conflict  580  (297 intervene / 283 quit)
                   Q_I mean -0.080 sd 0.996 ; Q_C mean -1.249 sd 0.141

PRIMARY: held_out_game, dR2 on Q_I of [obs+Q_C+act] over [obs+Q_C]
  cohort=all
    L8   +0.070 sd 0.021 same  [+0.036 +0.080 +0.075 +0.092 +0.065]
    L16  +0.113 sd 0.056 same  [+0.056 +0.131 +0.075 +0.200 +0.102]
    L24  +0.151 sd 0.060 same  [+0.104 +0.127 +0.160 +0.251 +0.113]
  cohort=original
    L8   +0.066 sd 0.039 same  [+0.028 +0.025 +0.094 +0.071 +0.112]
    L16  +0.128 sd 0.035 same  [+0.112 +0.132 +0.132 +0.181 +0.086]
    L24  +0.174 sd 0.072 same  [+0.080 +0.141 +0.274 +0.203 +0.172]
  ALL 30 PER-SPLIT VALUES POSITIVE. Monotone in depth in both cohorts.

ACTIVATIONS ALONE vs THE FULL BASELINE (held_out_game, R2 on Q_I)
  cohort=all       baseline obs+Q_C  0.403 [.539 .397 .389 .347 .341]
                   act_L24 alone     0.543 [.632 .511 .539 .593 .439]   5/5 higher
  cohort=original  baseline obs+Q_C  0.338 [.441 .368 .168 .376 .337]
                   act_L24 alone     0.505 [.509 .500 .441 .573 .501]   5/5 higher
  -> activations with NO observables and NO Q_C beat observables+Q_C on all 10 splits.

PRIMARY: held_out_game, intervene-vs-quit decision
  baseline obs+Q_C AUROC 0.899 (all) / 0.876 (original)  <- already near ceiling
  d-AUROC of [3] over [1]:
    all       L8 +0.017 same | L16 +0.021 same | L24 +0.029 FLIPS
    original  L8 +0.011 FLIPS | L16 +0.035 FLIPS | L24 +0.041 FLIPS
  d-macroF1: FLIPS in 5 of 6 held_out_game cells.
  act alone AUROC: all L24 0.923 vs baseline 0.899; original L24 0.915 vs 0.876.

SECONDARY: grouped_by_rollout (game channel open), cohort=all
  dR2  L8 +0.098 same | L16 +0.133 same | L24 +0.185 same
  d-AUROC L8 +0.022 same | L16 +0.033 same | L24 +0.040 same  (all stable here)

COMPANION (residualised, independent alphas) -- NOT the arbiter in this test
  all/held_out_game       L8 +0.027 FLIPS | L16 +0.099 FLIPS | L24 +0.198 same
  original/held_out_game  L8 -0.021 FLIPS | L16 -0.008 FLIPS | L24 +0.034 FLIPS
  In the original cohort the per-split values cluster at +-0.00x
  (e.g. L16 [+0.002 +0.003 -0.006 +0.007 -0.044]), the signature of ridge alpha
  collapsing to near-mean predictions on a noisy residual target at n_train~400.
  Not verified -- the companion's selected alphas are not printed.
  CRUCIALLY: the shared-alpha artifact the companion guards against biases model [3]
  DOWNWARD. Model [3] winning anyway means the artifact can only have made these
  numbers an underestimate. The companion was insurance against a spurious NEGATIVE,
  and no negative occurred.
