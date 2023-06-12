import torch


class Configs:
    def __init__(self, initial_values, triggered_function, type, nms_idx=0):
        self.old_configs = initial_values.copy()
        self.new_configs = initial_values.copy()
        self.triggered_function = triggered_function
        self.type = type
        self.nms_idx = nms_idx

    @property
    def configs(self):
        return self.old_configs

    @configs.setter
    def configs(self, new_values):
        self.old_configs = new_values.copy()
        if not self.equal_configs(self.old_configs, self.new_configs):
            if self.type == 0:
                print("Detecting bounding boxes...")
                self.triggered_function()
            else:
                print(f"Generating {self.old_configs[0]}...")
                self.triggered_function(self.old_configs[0], self.old_configs[1], self.old_configs[2], self.old_configs[3], self.old_configs[4],self.old_configs[5], self.old_configs[6], self.old_configs[7], self.old_configs[8])
        self.new_configs = new_values.copy()

    def equal_configs(self, list1, list2):
        if len(list1) != len(list2):
            return False

        for item1, item2 in zip(list1, list2):
            if isinstance(item1, torch.Tensor) and isinstance(item2, torch.Tensor):
                if not torch.equal(item1, item2):
                    return False
            else:
                if item1 != item2:
                    return False
        return True
