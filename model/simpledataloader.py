import torch

class SimpleDataLoader:
    def __init__(self, data, labels, batch_size=64, shuffle=True):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(data)
        self.num_batches = self.num_samples // batch_size
        self.indices = list(range(self.num_samples))

    def __iter__(self):
        if self.shuffle:
            torch.manual_seed(123)  
            torch.cuda.manual_seed(123)
            self.indices = torch.randperm(self.num_samples).tolist()

        for batch_start in range(0, self.num_samples, self.batch_size):
            batch_indices = self.indices[batch_start:batch_start + self.batch_size]
            batch_data = [(self.data[i], self.labels[i]) for i in batch_indices]
            yield self.collate_fn(batch_data)

    def collate_fn(self, batch_data):
        batch_inputs = torch.stack([sample[0][0] for sample in batch_data])
        batch_targets = torch.tensor([sample[1] for sample in batch_data], dtype=torch.long)
        return batch_inputs, batch_targets
