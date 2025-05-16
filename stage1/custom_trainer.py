import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
import torch.distributed.nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from trainer_headers import *
from grad_cache_custom import GradCacheCustom
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union

# grad_cache version
class ContrastiveTrainer(Trainer):
    def __init__(
        self,
        loss_fn=None,
        multi_pos=1,
        chunk_size=8,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.grad_cache = GradCacheCustom(
            models=[self.model, self.model],
            chunk_sizes=[chunk_size] * 2,
            loss_fn=self.loss_fn,
            multi_pos=multi_pos,
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Compute sentence embeddings for query and document
        query_embeddings = model.forward(
            {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
            }
        )['sentence_embedding']
        doc_embeddings = model.forward(
            {
                'input_ids': inputs['doc_input_ids'],
                'attention_mask': inputs['doc_attention_mask'],
            }
        )['sentence_embedding']
        
        # Gather embeddings across all processes for contrastive loss
        query_embeddings_new = torch.distributed.nn.all_gather(query_embeddings)
        doc_embeddings_new = torch.distributed.nn.all_gather(doc_embeddings)
        query_embeddings_new = torch.cat(query_embeddings_new, dim=0)
        doc_embeddings_new = torch.cat(doc_embeddings_new, dim=0)

        # Compute the contrastive loss
        loss = self.loss_fn(query_embeddings_new, doc_embeddings_new)

        # Optionally return intermediate outputs
        return (loss, (query_embeddings, doc_embeddings)) if return_outputs else loss
    
    # Evaluation step
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        # Get the device of the model
        device = next(model.parameters()).device

        # Move inputs to the model's device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Disable gradient computation
        with torch.no_grad():
            # Compute query embedding
            query_embeddings = model.forward(
                {
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask'],
                }
            )['sentence_embedding']

            n = inputs['doc_input_ids'].shape[1]
            doc_embeddings = None

            # Loop through each document in the list (dimension 1)
            for i in range(0, n):
                doc_embedding = model.forward(
                    {
                        'input_ids': inputs['doc_input_ids'][:, i],
                        'attention_mask': inputs['doc_attention_mask'][:, i],
                    }
                )['sentence_embedding'].unsqueeze(1)
                doc_embeddings = doc_embedding if doc_embeddings is None else torch.cat((doc_embeddings, doc_embedding), dim=1)

            # Return output format: (loss, predictions, labels)
            if prediction_loss_only:
                return (None, None, None)
            
            return None, \
                (query_embeddings, doc_embeddings, inputs['id'], inputs['doc_id']), \
                torch.zeros_like(inputs['id'])
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step with support for gradient caching.

        Args:
            model (nn.Module): The model being trained.
            inputs (dict): A batch of input data.
            num_items_in_batch (optional): The number of training examples in the batch.

        Returns:
            torch.Tensor: The computed training loss.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        # SageMaker model parallelism path
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        self.grad_cache.compute_loss_context_manager = self.compute_loss_context_manager

        # Empty cache conditionally at configured steps
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # Handle special optimizer requirements
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        # Mean-reduce loss in multi-GPU setting
        if self.args.n_gpu > 1:
            loss = loss.mean()

        # Setup backward function depending on backend
        if self.use_apex:
            def apex_backward(loss):
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            self.grad_cache.backward_fn = apex_backward
        else:
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            def accelerator_backward(loss):
                # Normalize loss across accumulation steps unless the model handles this internally
                if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                    loss = loss / self.args.gradient_accumulation_steps
                self.accelerator.backward(loss, **kwargs)
            self.grad_cache.backward_fn = accelerator_backward

        # Use gradient caching to compute and apply gradients
        return self.grad_cache.cache_step(inputs, num_items_in_batch=num_items_in_batch).detach()
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        # Call the base class's save_model method
        super().save_model(output_dir, _internal_call)

        model = self.model

        # Only save model on rank 0 in distributed setup
        if torch.distributed.get_rank() == 0:
            print(f"Rank {torch.distributed.get_rank()} saving model.")
            save_dir = os.path.join(output_dir, "sentence-transformer-checkpoint")
            os.makedirs(save_dir, exist_ok=True)
            # Save in SentenceTransformer-compatible format
            model.save(save_dir, safe_serialization=False)

        # Synchronize all processes to ensure save is complete
        torch.distributed.barrier()
