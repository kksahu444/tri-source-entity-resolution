# Transfer checks (work order T2), tag `kaggle-0927-1016`, rule `two_threshold`

Flag = test differs from dev by more than 10% relative (US/India). France has no dev counterpart; it is compared with the dev average of US and India.

Flags: France:a_share_records_bestp_ge_0.5, France:d_final_p_mean, France:d_final_p_q75, France:d_stageA_p_mean, France:d_stageA_p_q05, France:d_stageA_p_q25, France:d_stageA_p_q50, France:d_stageA_p_q75, France:e_cands_per_s1, France:e_share_records_in_a_list, India:a_share_records_bestp_ge_0.5, India:d_final_p_mean, India:d_final_p_q75, India:d_stageA_p_mean, India:d_stageA_p_q05, India:d_stageA_p_q25, India:d_stageA_p_q50, India:d_stageA_p_q75, India:e_cands_per_s1, India:e_share_records_in_a_list, US:a_share_records_bestp_ge_0.5, US:d_final_p_mean, US:d_final_p_q75, US:d_stageA_p_mean, US:d_stageA_p_q05, US:d_stageA_p_q50, US:d_stageA_p_q75, US:e_cands_per_s1, US:e_share_records_in_a_list


## France (dev reference: mean of US and India)

| Metric | Dev | Test | Rel. diff | Flag |
|---|---|---|---|---|
| a_share_records_bestp_ge_0.5 | 0.1605 | 0.6247 | +289.3% | FLAG |
| b_cands_p_ge_0.5_per_s1 | 3.3726 | 3.4561 | +2.5% |  |
| b_kept_over_p_ge_0.5 | 0.9867 | 0.9846 | -0.2% |  |
| b_links_kept_per_s1 | 3.3278 | 3.4029 | +2.3% |  |
| c_empty_list_rate | 0.0578 | 0.0521 | -9.9% |  |
| d_final_p_mean | 0.1306 | 0.1958 | +50.0% | FLAG |
| d_final_p_q05 | 0.0001 | 0.0001 | +0.0% |  |
| d_final_p_q25 | 0.0001 | 0.0001 | +0.0% |  |
| d_final_p_q50 | 0.0001 | 0.0001 | +0.0% |  |
| d_final_p_q75 | 0.0002 | 0.0099 | +6224.9% | FLAG |
| d_final_p_q95 | 0.9999 | 0.9999 | +0.0% |  |
| d_stageA_p_mean | 0.1305 | 0.1974 | +51.2% | FLAG |
| d_stageA_p_q05 | 0.0000 | 0.0000 | +203.7% | FLAG |
| d_stageA_p_q25 | 0.0000 | 0.0000 | +265.9% | FLAG |
| d_stageA_p_q50 | 0.0000 | 0.0000 | +1107.5% | FLAG |
| d_stageA_p_q75 | 0.0004 | 0.0131 | +3637.6% | FLAG |
| d_stageA_p_q95 | 0.9999 | 0.9999 | +0.0% |  |
| e_cands_per_s1 | 26.1317 | 17.8690 | -31.6% | FLAG |
| e_share_records_in_a_list | 0.5615 | 0.9969 | +77.5% | FLAG |

## India (dev reference: India)

| Metric | Dev | Test | Rel. diff | Flag |
|---|---|---|---|---|
| a_share_records_bestp_ge_0.5 | 0.1602 | 0.5829 | +263.9% | FLAG |
| b_cands_p_ge_0.5_per_s1 | 3.3657 | 3.3955 | +0.9% |  |
| b_kept_over_p_ge_0.5 | 0.9876 | 0.9830 | -0.5% |  |
| b_links_kept_per_s1 | 3.3240 | 3.3379 | +0.4% |  |
| c_empty_list_rate | 0.0578 | 0.0580 | +0.3% |  |
| d_final_p_mean | 0.1285 | 0.1862 | +44.9% | FLAG |
| d_final_p_q05 | 0.0001 | 0.0001 | +0.0% |  |
| d_final_p_q25 | 0.0001 | 0.0001 | +0.0% |  |
| d_final_p_q50 | 0.0001 | 0.0001 | +0.0% |  |
| d_final_p_q75 | 0.0002 | 0.0074 | +3386.5% | FLAG |
| d_final_p_q95 | 0.9999 | 0.9999 | +0.0% |  |
| d_stageA_p_mean | 0.1285 | 0.1862 | +44.9% | FLAG |
| d_stageA_p_q05 | 0.0000 | 0.0000 | +27.8% | FLAG |
| d_stageA_p_q25 | 0.0000 | 0.0000 | +28.8% | FLAG |
| d_stageA_p_q50 | 0.0000 | 0.0000 | +250.8% | FLAG |
| d_stageA_p_q75 | 0.0004 | 0.0079 | +2012.1% | FLAG |
| d_stageA_p_q95 | 0.9999 | 0.9999 | +0.0% |  |
| e_cands_per_s1 | 26.4845 | 18.4662 | -30.3% | FLAG |
| e_share_records_in_a_list | 0.5522 | 0.9941 | +80.0% | FLAG |

Dev true matched share of records: 0.1620

## US (dev reference: US)

| Metric | Dev | Test | Rel. diff | Flag |
|---|---|---|---|---|
| a_share_records_bestp_ge_0.5 | 0.1607 | 0.5900 | +267.0% | FLAG |
| b_cands_p_ge_0.5_per_s1 | 3.3795 | 3.3968 | +0.5% |  |
| b_kept_over_p_ge_0.5 | 0.9858 | 0.9862 | +0.0% |  |
| b_links_kept_per_s1 | 3.3316 | 3.3498 | +0.5% |  |
| c_empty_list_rate | 0.0577 | 0.0572 | -1.0% |  |
| d_final_p_mean | 0.1326 | 0.1883 | +42.0% | FLAG |
| d_final_p_q05 | 0.0001 | 0.0001 | +0.0% |  |
| d_final_p_q25 | 0.0001 | 0.0001 | +0.0% |  |
| d_final_p_q50 | 0.0001 | 0.0001 | +0.0% |  |
| d_final_p_q75 | 0.0001 | 0.0035 | +3415.8% | FLAG |
| d_final_p_q95 | 0.9999 | 0.9999 | +0.0% |  |
| d_stageA_p_mean | 0.1325 | 0.1882 | +42.0% | FLAG |
| d_stageA_p_q05 | 0.0000 | 0.0000 | +15.3% | FLAG |
| d_stageA_p_q25 | 0.0000 | 0.0000 | +0.2% |  |
| d_stageA_p_q50 | 0.0000 | 0.0000 | +152.2% | FLAG |
| d_stageA_p_q75 | 0.0003 | 0.0059 | +1702.0% | FLAG |
| d_stageA_p_q95 | 0.9999 | 0.9999 | +0.0% |  |
| e_cands_per_s1 | 25.7789 | 18.2117 | -29.4% | FLAG |
| e_share_records_in_a_list | 0.5709 | 0.9898 | +73.4% | FLAG |

Dev true matched share of records: 0.1626
