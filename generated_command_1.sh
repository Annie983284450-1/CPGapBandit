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

 
