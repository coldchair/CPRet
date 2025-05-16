import torch
from sentence_transformers import SentenceTransformerTrainer
from torch.utils.data import ConcatDataset
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, Value
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformerTrainer

class CustomTrainer(SentenceTransformerTrainer):
    def __init__(
        self,
        batch_sizes=None,  # New parameter: dictionary specifying batch sizes for each sub-dataset
        *args,
        **kwargs
    ):
        self.batch_sizes = batch_sizes or {}
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        if isinstance(self.train_dataset, DatasetDict):
            generator = torch.Generator()
            if self.args.seed:
                generator.manual_seed(self.args.seed)

            data_collator = self.data_collator
            batch_samplers = []

            # 👇 Key logic: iterate over (name, dataset) pairs, assign batch size based on name
            for name, dataset in self.train_dataset.items():
                bs = self.batch_sizes.get(name, self.args.train_batch_size)
                sampler = self.get_batch_sampler(
                    dataset,
                    batch_size=bs,
                    drop_last=self.args.dataloader_drop_last,
                    valid_label_columns=data_collator.valid_label_columns,
                    generator=generator,
                )
                batch_samplers.append(sampler)

            concat = ConcatDataset(self.train_dataset.values())
            batch_sampler = self.get_multi_dataset_batch_sampler(
                dataset=concat,
                batch_samplers=batch_samplers,
                generator=generator,
                seed=self.args.seed,
            )

            dl_kwargs = dict(
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                persistent_workers=self.args.dataloader_persistent_workers,
                prefetch_factor=self.args.dataloader_prefetch_factor,
                batch_sampler=batch_sampler,
            )

            self.accelerator.even_batches = False
            self._train_dataloader = self.accelerator.prepare(DataLoader(concat, **dl_kwargs))
            return self._train_dataloader

        # For single dataset or IterableDataset, fall back to default behavior
        return super().get_train_dataloader()
