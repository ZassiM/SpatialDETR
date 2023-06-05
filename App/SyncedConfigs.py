class SyncedConfigs:
    def __init__(self, initial_values, triggered_function, type):
        self.old_configs = initial_values.copy()
        self.new_configs = initial_values.copy()
        self.triggered_function = triggered_function
        self.type = type

    @property
    def configs(self):
        return self.old_configs

    @configs.setter
    def configs(self, new_values):
        self.old_configs = new_values.copy()
        if self.old_configs != self.new_configs:
            if self.type == 0:
                print("Detecting bounding boxes...")
                self.triggered_function()
            else:
                print("Generating explainability maps...")
                self.attn_list = self.triggered_function(self.old_configs[0], self.old_configs[1], self.old_configs[2], self.old_configs[3], self.old_configs[4],self.old_configs[5], self.old_configs[6], self.old_configs[7])
        self.new_configs = new_values.copy()
