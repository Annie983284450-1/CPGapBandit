nohup python cpbandit_reward_upated.py enet dct --num_test_sepsis_pat_win 100 --num_train_sepsis_pat_win 5000 --num_train_nosepsis_pat_win 5000 > ../cpbanditsepsis_experiements/no_refit_balanced_win8_new_reward_regret/Results_100_5000_enet_dct/v2regret100_5000_enet_dct.log 2>&1 &
nohup python cpbandit_reward_upated.py enet dct rf --num_test_sepsis_pat_win 100 --num_train_sepsis_pat_win 5000 --num_train_nosepsis_pat_win 5000 > ../cpbanditsepsis_experiements/no_refit_balanced_win8_new_reward_regret/Results_100_5000_enet_dct_rf/v2regret100_5000_enet_dct_rf.log 2>&1 &
nohup python cpbandit_reward_upated.py enet dct rf svr --num_test_sepsis_pat_win 100 --num_train_sepsis_pat_win 5000 --num_train_nosepsis_pat_win 5000 > ../cpbanditsepsis_experiements/no_refit_balanced_win8_new_reward_regret/Results_100_5000_enet_dct_rf_svr/v2regret100_5000_enet_dct_rf_svr.log 2>&1 &
nohup python cpbandit_reward_upated.py enet dct --num_test_sepsis_pat_win 10 --num_train_sepsis_pat_win 5 --num_train_nosepsis_pat_win 5 > ../cpbanditsepsis_experiements/no_refit_balanced_win8_new_reward_regret/Results_100_5000_enet_dct/v2regret100_5000_enet_dct.log 2>&1 &



nohup python cpbandit_prob.py > nohup_prob.log 2>&1 &
nohup python cpbandit_prob.py > nohup_prob_ridge_rf_dct_lr_svr300_1000.log 2>&1 &
nohup python cpbandit_prob.py > nohup_prob_ridge_rf_dct_lr_svr_xgb300_1000.log 2>&1 &


~~~~~~~~~~~~~~~~~~~~ Run on 2024.10.09 10:40AM WED ~~~~~~~~~~~~~~~~~~~~~~~~
jobID:1542365
nohup python cpbandit_prob.py > nohup_prob_ridge_rf_dct_lr_svr_lgb300_1000.log 2>&1 &

## wrong. cannot load the previous model properly, since the previous models is not dumped properly in the right folder

## testing the new code for properly loading the previous model
nohup python cpbandit_prob.py > nohup_prob_ridge_rf_dct_lr_svr_lgb10_20_refitstep_10.log 2>&1 &
nohup python cpbandit_prob.py > nohup_prob_lasso_combo_300_1000_refitstep_100.log 2>&1 &
nohup python cpbandit_prob.py > nohup_prob_1lgb_2xgb_3cat_combo_300_1000_refitstep_100.log 2>&1 &
nohup python cpbandit_prob.py --num > nohup_prob_testSeptic2_trainSeptic5_refitstep_1.log 2>&1 &

# parser.add_argument('--num_test_sepsis_pat', type=int, default=2)
# parser.add_argument('--num_train_sepsis_pat', type=int, default=5)
# parser.add_argument('--refit_step', type=int, default=1)
# parser.add_argument('--B', type=int, default=25)
nohup python cpbandit_prob.py --num_test_sepsis_pat 100 --num_train_sepsis_pat 2000 > nohup_prob_testSeptic100_trainSeptic2000_NoRefit_lassocombo.log 2>&1 &

nohup python cpbandit_prob.py --num_test_sepsis_pat 50 --num_train_sepsis_pat 1000 > nohup_prob_testSeptic50_trainSeptic1000_NoRefit_nnetcombo.log 2>&1 &
nohup python cpbandit_prob.py --num_test_sepsis_pat 100 --num_train_sepsis_pat 2000 --combo lgb > nohup_prob_testSeptic100_trainSeptic2000_NoRefit_lgbcombo.log 2>&1 &