from Models.SarsaZeroAfterstates import SarsaZeroAfterStates

model = SarsaZeroAfterStates.load(r'D:\Bibliotheken\Downloads\model.pickle')
model.train(lambda x: 1/x, 10)
