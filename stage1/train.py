import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import TrainingArguments
from loss import InfoNCELoss_gradcache, InfoNCELoss_gradcache_multipos, GroupInfoNCELoss_gradcache_multipos
from dataset import ContrastiveDataset, ContrastiveDataCollator
from custom_trainer import ContrastiveTrainer
from eval_metric import compute_metrics_custom
from dataset_process import process_dataset

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str,
                    default="Salesforce/SFR-Embedding-Code-2B_R")
parser.add_argument("--max_length", type=int, default=1024)
parser.add_argument("--chunk_size", type=int, default=2)
parser.add_argument("--multi_pos", type=int, default=16)
parser.add_argument("--per_device_batch_size", type=int, default=128)
parser.add_argument("--num_train_epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=3e-8)
parser.add_argument("--temperature", type=float, default=0.07)
parser.add_argument("--loss_type", type=str, choices=["GroupInfoNCE", "InfoNCE", "InfoNCE_multipos"],
                    default="GroupInfoNCE")
parser.add_argument("--use_data_augmentation", type=str2bool, default=True)
parser.add_argument("--eval_only", type=str2bool, default=False)

if __name__ == "__main__":
    args = parser.parse_args()

    model_name = args.model_path.split('/')[-1]

    # Load dataset from HuggingFace
    data = load_dataset('coldchair16/CPRet-data', 'PCPCD')
    traindata, testdata = process_dataset(data, args.max_length)

    if model_name == 'SFR-Embedding-Code-2B_R' or model_name.startswith('Qwen3-Embedding'):
        model_kwargs = {"torch_dtype": torch.bfloat16}
    else:
        model_kwargs = {}
        
    model = SentenceTransformer(args.model_path, trust_remote_code=True, model_kwargs=model_kwargs, device='cpu')
    model.tokenizer.model_max_length = args.max_length
    model.max_seq_length = args.max_length

    dataset = ContrastiveDataset(
        traindata, model.tokenizer, max_length=args.max_length, multi_pos=args.multi_pos,
        use_data_augmentation=args.use_data_augmentation,
    )
    val_dataset = ContrastiveDataset(
        testdata, model.tokenizer, max_length=args.max_length, multi_pos=args.multi_pos,
        use_data_augmentation=False,
        eval_mode=True,
    )

    embedding_dim = model.get_sentence_embedding_dimension()
    print("Model embedding dimension: ", embedding_dim)

    if args.loss_type == 'GroupInfoNCE':
        loss_fn = GroupInfoNCELoss_gradcache_multipos(temperature=args.temperature)
    elif args.loss_type == 'InfoNCE':
        loss_fn = InfoNCELoss_gradcache(temperature=args.temperature)
    elif args.loss_type == 'InfoNCE_multipos':
        loss_fn = InfoNCELoss_gradcache_multipos(temperature=args.temperature)

    data_collator = ContrastiveDataCollator(model.tokenizer)

    save_name = args.model_path.split('/')[-1]
    save_name += f"-length{args.max_length}"
    save_name += f"-bs_per_device{args.per_device_batch_size}*gpus{os.environ['WORLD_SIZE']}"
    save_name += f"-epochs{args.num_train_epochs}"
    save_name += f"-temperature{args.temperature}"
    save_name += f"-lr{args.lr}"
    save_name += f"-multi_pos{args.multi_pos}"
    save_name += f"-use_data_augmentation_{args.use_data_augmentation}"
    save_name += f"-loss{args.loss_type}"
    
    print(f"Save name: {save_name}")

    training_args = TrainingArguments(
        output_dir="./results/" + save_name,
        per_device_train_batch_size=args.per_device_batch_size,
        num_train_epochs=args.num_train_epochs,
        logging_dir="./logs/" + save_name,
        logging_steps=1,
        eval_strategy="epoch",
        # eval_accumulation_steps=5,
        save_strategy="epoch",
        # save_steps=1,
        bf16=True,
        dataloader_num_workers=4,
        dataloader_drop_last=True,
        save_total_limit=1,
        remove_unused_columns=False,
        learning_rate=args.lr,
        weight_decay=0.01,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        lr_scheduler_type="constant",
    )

    trainer = ContrastiveTrainer(
        multi_pos=args.multi_pos,
        chunk_size=args.chunk_size,
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        loss_fn=loss_fn,
        data_collator=data_collator,
        compute_metrics=compute_metrics_custom,
    )

    if args.eval_only:
        eval_results = trainer.evaluate()
        print(eval_results)
    else:
        trainer.train()
