# Transfer checks (work order T2), tag `kaggle-0926-1219`, rule `gfm_indep`

Flag = test differs from dev by more than 10% relative (US/India). France has no dev counterpart; it is compared with the dev average of US and India.

Flags: France:a_share_records_bestp_ge_0.5, France:b_cands_p_ge_0.5_per_s1, France:c_empty_list_rate, France:d_final_p_mean, France:d_final_p_q75, France:d_stageA_p_mean, France:d_stageA_p_q05, France:d_stageA_p_q25, France:d_stageA_p_q50, France:d_stageA_p_q75, France:e_cands_per_s1, France:e_share_records_in_a_list, India:a_share_records_bestp_ge_0.5, India:d_final_p_mean, India:d_final_p_q75, India:d_stageA_p_mean, India:d_stageA_p_q50, India:d_stageA_p_q75, India:e_cands_per_s1, India:e_share_records_in_a_list, US:a_share_records_bestp_ge_0.5, US:b_cands_p_ge_0.5_per_s1, US:b_links_kept_per_s1, US:d_final_p_mean, US:d_final_p_q75, US:d_stageA_p_mean, US:d_stageA_p_q25, US:d_stageA_p_q50, US:d_stageA_p_q75, US:e_cands_per_s1, US:e_share_records_in_a_list


## France (dev reference: mean of US and India)

| Metric | Dev | Test | Rel. diff | Flag |
|---|---|---|---|---|
| a_share_records_bestp_ge_0.5 | 0.1637 | 0.6826 | +317.1% | FLAG |
| b_cands_p_ge_0.5_per_s1 | 3.3640 | 3.7762 | +12.3% | FLAG |
| b_kept_over_p_ge_0.5 | 0.9850 | 0.9425 | -4.3% |  |
| b_links_kept_per_s1 | 3.3134 | 3.5593 | +7.4% |  |
| c_empty_list_rate | 0.0579 | 0.0506 | -12.7% | FLAG |
| d_final_p_mean | 0.1463 | 0.2440 | +66.8% | FLAG |
| d_final_p_q05 | 0.0001 | 0.0001 | +0.0% |  |
| d_final_p_q25 | 0.0001 | 0.0001 | +0.0% |  |
| d_final_p_q50 | 0.0001 | 0.0001 | +0.0% |  |
| d_final_p_q75 | 0.0001 | 0.3738 | +309922.9% | FLAG |
| d_final_p_q95 | 0.9999 | 0.9999 | +0.0% |  |
| d_stageA_p_mean | 0.1462 | 0.2351 | +60.8% | FLAG |
| d_stageA_p_q05 | 0.0000 | 0.0000 | +156.3% | FLAG |
| d_stageA_p_q25 | 0.0000 | 0.0000 | +159.3% | FLAG |
| d_stageA_p_q50 | 0.0000 | 0.0000 | +734.9% | FLAG |
| d_stageA_p_q75 | 0.0003 | 0.1908 | +57702.6% | FLAG |
| d_stageA_p_q95 | 1.0000 | 1.0000 | +0.0% |  |
| e_cands_per_s1 | 23.2078 | 15.6397 | -32.6% | FLAG |
| e_share_records_in_a_list | 0.5328 | 0.9690 | +81.9% | FLAG |

## India (dev reference: India)

| Metric | Dev | Test | Rel. diff | Flag |
|---|---|---|---|---|
| a_share_records_bestp_ge_0.5 | 0.1605 | 0.6061 | +277.6% | FLAG |
| b_cands_p_ge_0.5_per_s1 | 3.3497 | 3.5311 | +5.4% |  |
| b_kept_over_p_ge_0.5 | 0.9843 | 0.9621 | -2.3% |  |
| b_links_kept_per_s1 | 3.2970 | 3.3971 | +3.0% |  |
| c_empty_list_rate | 0.0585 | 0.0581 | -0.6% |  |
| d_final_p_mean | 0.1455 | 0.2220 | +52.6% | FLAG |
| d_final_p_q05 | 0.0001 | 0.0001 | +0.0% |  |
| d_final_p_q25 | 0.0001 | 0.0001 | +0.0% |  |
| d_final_p_q50 | 0.0001 | 0.0001 | +0.0% |  |
| d_final_p_q75 | 0.0001 | 0.0599 | +42356.9% | FLAG |
| d_final_p_q95 | 0.9999 | 0.9999 | +0.0% |  |
| d_stageA_p_mean | 0.1454 | 0.2165 | +48.9% | FLAG |
| d_stageA_p_q05 | 0.0000 | 0.0000 | +9.6% |  |
| d_stageA_p_q25 | 0.0000 | 0.0000 | -4.1% |  |
| d_stageA_p_q50 | 0.0000 | 0.0000 | +45.4% | FLAG |
| d_stageA_p_q75 | 0.0004 | 0.0392 | +9508.2% | FLAG |
| d_stageA_p_q95 | 1.0000 | 1.0000 | +0.0% |  |
| e_cands_per_s1 | 23.2301 | 16.0777 | -30.8% | FLAG |
| e_share_records_in_a_list | 0.5171 | 0.9706 | +87.7% | FLAG |

Dev true matched share of records: 0.1619

## US (dev reference: US)

| Metric | Dev | Test | Rel. diff | Flag |
|---|---|---|---|---|
| a_share_records_bestp_ge_0.5 | 0.1668 | 0.6918 | +314.8% | FLAG |
| b_cands_p_ge_0.5_per_s1 | 3.3783 | 3.9830 | +17.9% | FLAG |
| b_kept_over_p_ge_0.5 | 0.9857 | 0.9310 | -5.5% |  |
| b_links_kept_per_s1 | 3.3298 | 3.7084 | +11.4% | FLAG |
| c_empty_list_rate | 0.0574 | 0.0521 | -9.1% |  |
| d_final_p_mean | 0.1472 | 0.2528 | +71.8% | FLAG |
| d_final_p_q05 | 0.0001 | 0.0001 | +0.0% |  |
| d_final_p_q25 | 0.0001 | 0.0001 | +0.0% |  |
| d_final_p_q50 | 0.0001 | 0.0001 | +0.0% |  |
| d_final_p_q75 | 0.0001 | 0.5127 | +512620.2% | FLAG |
| d_final_p_q95 | 0.9999 | 0.9999 | +0.0% |  |
| d_stageA_p_mean | 0.1471 | 0.2371 | +61.2% | FLAG |
| d_stageA_p_q05 | 0.0000 | 0.0000 | -9.8% |  |
| d_stageA_p_q25 | 0.0000 | 0.0000 | -28.3% | FLAG |
| d_stageA_p_q50 | 0.0000 | 0.0000 | -14.9% | FLAG |
| d_stageA_p_q75 | 0.0003 | 0.2444 | +96831.0% | FLAG |
| d_stageA_p_q95 | 1.0000 | 1.0000 | +0.0% |  |
| e_cands_per_s1 | 23.1856 | 15.8543 | -31.6% | FLAG |
| e_share_records_in_a_list | 0.5485 | 0.9836 | +79.3% | FLAG |

Dev true matched share of records: 0.1684
