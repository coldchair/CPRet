import torch
from torch.utils.data import Dataset, DataLoader
import random
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding


# Define the ContrastiveDataset
class ContrastiveDataset(Dataset):
    def __init__(
        self,
        data, tokenizer,
        max_length = 1024,
        eval_mode = False,
        prompt_question = "",
        prompt_code = "",
        eval_max_solutions_per_question = 20,
        multi_pos = 1,
        use_data_augmentation = True,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eval_mode = eval_mode
        self.prompt_question = prompt_question
        self.prompt_code = prompt_code
        self.multi_pos = multi_pos
        self.use_data_augmentation = use_data_augmentation

        self.eval_max_solutions_per_question = eval_max_solutions_per_question

        self.len = len(self.data)

        # for debug
        # if (eval_mode):
        #     self.len = 8

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if (self.eval_mode or not self.use_data_augmentation):
            query = self.data[idx]['question']
        else:
            query = random.choice([
                self.data[idx]['question'],
                self.data[idx]['question-main'],
            ])
                
        query = self.prompt_question + query
        query_encoding = self.tokenizer([query], return_tensors="pt", padding='max_length', max_length=self.max_length, truncation=True)

        if (self.eval_mode):
            n = min(self.eval_max_solutions_per_question, len(self.data[idx]['solutions']))
            doc = self.data[idx]['solutions'][:n]
            doc = [self.prompt_code + d for d in doc]
            if (n < self.eval_max_solutions_per_question):
                doc = doc + [""] * (self.eval_max_solutions_per_question - n)
            doc_encoding = self.tokenizer(doc, return_tensors="pt", padding='max_length', max_length=self.max_length, truncation=True)
            doc_id = torch.tensor(n)
        else:
            if (self.multi_pos == 1):
                doc = random.choice(self.data[idx]['solutions'])
                doc = self.prompt_code + doc
                doc_encoding = self.tokenizer([doc], return_tensors="pt", padding='max_length', max_length=self.max_length, truncation=True)
                doc_encoding['input_ids'] = doc_encoding['input_ids'].squeeze(0)
                doc_encoding['attention_mask'] = doc_encoding['attention_mask'].squeeze(0)
                doc_id = torch.tensor(1)
            else:
                n = len(self.data[idx]['solutions'])
                if (n < self.multi_pos):
                    doc = self.data[idx]['solutions'] * (self.multi_pos // n) + self.data[idx]['solutions'][:self.multi_pos % n]
                else:
                    doc = random.sample(self.data[idx]['solutions'], self.multi_pos)
                doc = [self.prompt_code + d for d in doc]
                doc_encoding = self.tokenizer(doc, return_tensors="pt", padding='max_length', max_length=self.max_length, truncation=True)
                doc_id = torch.tensor(n)
        
        # Return a dictionary with the required fields
        return {
            # shape (max_length)
            'input_ids': query_encoding['input_ids'].squeeze(0),  # Standard field for Trainer
            'attention_mask': query_encoding['attention_mask'].squeeze(0),  # Standard field for Trainer
            'id' : torch.tensor(idx),

            # shape : (1, max_length) or (n, max_length)
            'doc_input_ids': doc_encoding['input_ids'],  # Additional field for document
            'doc_attention_mask': doc_encoding['attention_mask'],  # Additional field for document
            'doc_id': doc_id,  # Additional field for evaluation
        }

# Custom DataCollator to handle additional fields
from transformers import DataCollator

class ContrastiveDataCollator:
    def __init__(self, tokenizer, padding=True, truncation=True, max_length=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length

    def __call__(self, features):
        batch = {}
        if 'input_ids' in features[0]:
            batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
        if 'attention_mask' in features[0]:
            batch['attention_mask'] = torch.stack([f['attention_mask'] for f in features])
        if 'id' in features[0]:
            batch['id'] = torch.stack([f['id'] for f in features])

        if 'doc_input_ids' in features[0]:
            batch['doc_input_ids'] = torch.stack([f['doc_input_ids'] for f in features])
        if 'doc_attention_mask' in features[0]:
            batch['doc_attention_mask'] = torch.stack([f['doc_attention_mask'] for f in features])
        if 'doc_id' in features[0]:
            batch['doc_id'] = torch.stack([f['doc_id'] for f in features])
        return batch