from Evaluation.Evaluate_policy import evaluate_policy
from Models.SarsaZeroAfterstates import SarsaZeroAfterStates


model = SarsaZeroAfterStates(0.05, 0.9)
model.train(lambda x: 1/(1+x), 10)
