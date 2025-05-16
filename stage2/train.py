import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainingArguments,
)
from datasets import load_dataset, Dataset
from transformers.integrations import TensorBoardCallback
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import TripletLoss, CachedMultipleNegativesRankingLoss
from sentence_transformers.losses import TripletDistanceMetric
from trainer_custom import CustomTrainer
from sentence_transformers.training_args import BatchSamplers
import argparse
from dataset_process import process_dataset, process_eval_dataset
from sentence_transformers.training_args import MultiDatasetBatchSamplers
import random

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Initialize argument parser
parser = argparse.ArgumentParser(description='Fine-tune a model for sentence similarity')

# Add arguments
parser.add_argument('--model_path',
                    default='coldchair16/CPRetriever-Code',
                    type=str,
                    help='The name of the model to fine-tune')

parser.add_argument('--triplet_margin',
                    default=0.3,
                    type=float,
                    help='The margin for triplet loss')

parser.add_argument('--cut',
                    default=True,
                    type=str2bool,
                    help='Whether to cut the dataset SimplifiedRetrieval')

parser.add_argument('--lr',
                    default=2e-6,
                    type=float,
                    help='The learning rate for training')

parser.add_argument('--epochs',
                    default=1,
                    type=int,
                    help='Number of epochs for training')

parser.add_argument('--dataset_id',
                    default='coldchair16/CPRet-data',
                    type=str,
                    help='The dataset ID for training')

parser.add_argument('--eval_task_list',
                    default=['T2C', 'C2C', 'P2Dup', 'S2Full'],
                    type=str,
                    nargs='+',
                    help='List of evaluation tasks')

parser.add_argument('--train_task_list',
                    default=['P2Dup', 'S2Full'],
                    # default=['PCD', 'P2Dup', 'S2Full'],
                    type=str,
                    nargs='+',
                    help="List of training tasks: ['PCD', 'P2Dup', 'S2Full']")

parser.add_argument('--max_length',
                    default=1024,
                    type=int,
                    help='Maximum input length for the model')

parser.add_argument('--eval_only',
                    default=False,
                    type=str2bool,
                    help='Whether to run evaluation only, not training')


def main():
    # Parse arguments
    args = parser.parse_args()
    # Use the parsed arguments in your code
    dataset_id = args.dataset_id
    eval_task_list = args.eval_task_list
    train_task_list = args.train_task_list
    max_length = args.max_length
    eval_only = args.eval_only
    model_path = args.model_path
    triplet_margin = args.triplet_margin
    cut = args.cut
    lr = args.lr
    epochs = args.epochs

    model_name = model_path.split('/')[-1]

    if model_name in ['SFR-Embedding-Code-2B_R', 'CPRetriever-Code', 'CPRetriever-Prob']:
        model_kwargs = {"torch_dtype": torch.bfloat16}
    else:
        model_kwargs = {}
    model = SentenceTransformer(model_path, trust_remote_code=True, model_kwargs=model_kwargs)

    model.tokenizer.model_max_length = max_length
    model.max_seq_length = max_length

    evaluator_list = []
    for task in eval_task_list:
        queries = load_dataset(dataset_id, f"{task}-queries", split='test')
        corpus = load_dataset(dataset_id, f"{task}-corpus", split='test')
        qrels = load_dataset(dataset_id, f"{task}-qrels", split='test')
        queries, corpus, relevant_docs = process_eval_dataset(queries, corpus, qrels)
        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=task,
        )
        evaluator_list.append(evaluator)
    
    train_dataset = {}
    for task in train_task_list:
        if (task == 'PCD'):
            data = load_dataset(dataset_id, f"PCPCD")
            traindata_PCPCD, testdata_PCPCD = process_dataset(data, max_length)
            anchor = []
            positive = []
            for d in traindata_PCPCD:
                anchor.append(d['question'])
                positive.append(random.choice(d['solutions']))
            dataset = Dataset.from_dict({
                'anchor' : anchor,
                'positive' : positive,
            })
        elif (task == 'P2Dup' or task == 'S2Full'):
            data = load_dataset(dataset_id, f"{task}-train-pairs", split='train')
            dataset = Dataset.from_dict({
                'anchor': data['query'],
                'positive': data['pos'],
                'negative': data['neg'],
            })
        if (cut and task == 'S2Full'):
            dataset = dataset.select(random.sample(range(len(dataset)), 1000))
        print(f"train task = {task} len = {len(dataset)}")
        train_dataset[task] = dataset


    output_dir = f"./results/{model_name}"
    output_dir = output_dir + '_' + '-'.join(train_task_list)
    output_dir = output_dir + f"_margin{triplet_margin}"
    output_dir = output_dir + f"_lr{lr}"
    output_dir = output_dir + f'_maxlength{max_length}'
    if (cut):
        output_dir = output_dir + "_cut"
    output_dir = output_dir + f"_epochs{epochs}"
    print(f"output_dir : {output_dir}")


    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=lr,
        lr_scheduler_type='constant',
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=10,
        run_name='CPRet',  # Will be used in W&B if `wandb` is installed
        save_on_each_node=False,
        max_grad_norm=1.0,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
    )


    losses = {
        'PCD' : CachedMultipleNegativesRankingLoss(model, mini_batch_size=2, scale=14.28571429),
        'P2Dup' : TripletLoss(model, TripletDistanceMetric.COSINE, triplet_margin=triplet_margin),
        'S2Full' : TripletLoss(model, TripletDistanceMetric.COSINE, triplet_margin=triplet_margin),
    }

    # 为每个数据集单独设置 batch size
    batch_sizes = {
        'PCD' : 32,
        'P2Dup' : 1,
        'S2Full' : 1,
    }
    
    trainer = CustomTrainer(
        batch_sizes=batch_sizes,
        model=model,
        train_dataset=train_dataset,
        args = args,
        loss=losses,
        callbacks=[TensorBoardCallback()],
        evaluator=evaluator_list,
    )
    if (eval_only):
        eval_results = trainer.evaluate()
        print(eval_results)
    else:
        trainer.train()

if __name__ == '__main__':
    main()
