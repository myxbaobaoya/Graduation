import train_model
import numpy as np
import torch

class try_update():
    def Try(self,model_number):
        self.train_temp = []
        with open("update_train.txt") as tdata:
            while True:
                line = tdata.readline()
                if not line:
                    break
                self.train_temp.append([float(i) for i in line.split()])
        self.train_temp = np.array(self.train_temp)

        self.result_temp = []
        with open("update_result.txt") as rdata:
            while True:
                line = rdata.readline()
                if not line:
                    break
                self.result_temp.append([float(i) for i in line.split()])
        self.result_temp = np.array(self.result_temp)


        self.y = np.mat(self.result_temp)
        self.y = torch.tensor(self.y).float()
        run = train_model.train()
        run.update(model_number,self.train_temp,self.y)


# go = try_update()
# go.Try(1)
# go.Try(2)