from Models.SarsaZeroAfterstates import SarsaZeroAfterStates
from Evaluation.render_policy import render_policy
model = SarsaZeroAfterStates.load(r'C:\Users\xande\Downloads\model.pickle')
render_policy(model, 10)
