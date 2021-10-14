from Models.OnPolicyMCForTetris import OnPolicyMCForTetris
from Evaluation.Evaluate_policy import evaluate_policy_state_action
from Evaluation.Render_policy import render_policy_state_action
from tetris_environment.tetris_env import TetrisEnv

model = OnPolicyMCForTetris.load(r"C:\Users\xande\Downloads\model.pickle")
metrics = evaluate_policy_state_action(model, model.env, 500)
print(metrics.mean)
print(metrics.quantile([0.25, 0.5, 0.75]))
model.env = TetrisEnv(type="fourer", render=True)
render_policy_state_action(model, model.env, 10)