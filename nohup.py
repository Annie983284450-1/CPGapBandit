import subprocess
import os
def generate_command(num_test_pat_win, num_train_sepsis_pat_win, num_train_nosepsis_pat_win, expert_args, output_file):
    command = (
        f"nohup python cpbandit_reward_upated.py {expert_args} "
        f"--num_test_pat_win {num_test_pat_win} "
        f"--num_train_sepsis_pat_win {num_train_sepsis_pat_win} "
        f"--num_train_nosepsis_pat_win {num_train_nosepsis_pat_win} "
        f"> {output_file} 2>&1 &"
    )

    return command
    # subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Example usage
num_test_pat_win = 100
num_train_sepsis_pat_win = 5000
num_train_nosepsis_pat_win = 5000
experts = ["rf", "xgb", "ridge", "lasso",  
           "rf ridge", "rf lasso", "rf xgb", "xgb lasso", "ridge xgb", "ridge lasso",
           "xgb lasso ridge", "rf xgb ridge", "rf lasso ridge", "rf xgb lasso", 
           "rf xgb lasso ridge"]

experts = ['svr', 'enet', 'dct']
experts = ['enet dct', 'enet dct rf', 'enet dct rf svr'] 

win_size = 8
i=0
for expert in experts:
    
    expert_list = str(expert.replace(' ', '_'))
    # final_result_path = f'../cpbanditsepsis_experiements/no_refit_balanced_win{win_size}_updatedreward/Results_{str(num_test_pat_win)}_{str(num_train_sepsis_pat_win)}_{expert_list}' 
    final_result_path = f'../cpbanditsepsis_experiements/no_refit_balanced_win{win_size}_new_reward_regret/Results_{str(num_test_pat_win)}_{str(num_train_sepsis_pat_win)}_{expert_list}' 

    if not os.path.exists(final_result_path):
        os.makedirs(final_result_path)
    output_file = final_result_path + f"/v2regret{num_test_pat_win}_{num_train_sepsis_pat_win}_{expert_list}.log"
    command_string = generate_command(num_test_pat_win, num_train_sepsis_pat_win, num_train_nosepsis_pat_win, expert, output_file)
    
    if i ==0:
        with open("generated_command_1.sh", "w") as file:
            file.write(command_string + "\n")
    else:
        with open("generated_command_1.sh", "a") as file:
            file.write(command_string + "\n")
    i+=1