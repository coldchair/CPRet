from grad_cache import GradCache
from typing import List, Union, Callable, Any
from contextlib import nullcontext
import torch
from torch import nn, Tensor
from torch.cuda.amp import GradScaler, autocast
from grad_cache.context_managers import RandContext
    
class GradCacheCustom(GradCache):
    def __init__(
        self,
        multi_pos : int = 1,
        backward_fn = None,
        compute_loss_context_manager = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.multi_pos = multi_pos
        self.backward_fn = backward_fn
        self.compute_loss_context_manager = None
        self.get_rep_fn = lambda x: x['sentence_embedding']
    
    def model_call(self, model: nn.Module, model_input):
        """
        Literally call the model's __call__ method.
        :param model: model to be called
        :param model_input: input to the model call
        :return: model output
        """
        with self.compute_loss_context_manager():
            return model(model_input)

    def compute_loss(self, *reps: Tensor, **loss_kwargs) -> Tensor:
        """
        Compute the loss based on the representation tensors. The tensors should be ordered same as the list of models
        registered in this GradCache class instance.
        :param reps: Representations for computing the loss.
        :param loss_kwargs: Keyword arguments input to the loss function.
        :return: the loss tensor.
        """
        with self.compute_loss_context_manager():
            loss = self.loss_fn(*reps, **loss_kwargs)
        return loss
    
    def build_cache(self, *reps: Tensor, **loss_kwargs) -> [List[Tensor], Tensor]:
        """
        Compute the gradient cache
        :param reps: Computed representations from all encoder models
        :param loss_kwargs: Extra keyword arguments to the loss function
        :return: A tuple of a) gradient cache for each encoder model, and b) loss tensor
        """
        # for r in reps:
        #     print(f"r.shape : {r.shape}")
        reps = [r.detach().requires_grad_() for r in reps]
        with autocast() if self.fp16 else nullcontext():
            loss = self.compute_loss(*reps, **loss_kwargs)

        if self.fp16:
            self.backward_fn(self.scaler.scale(loss))
        else:
            self.backward_fn(loss)

        cache = [r.grad for r in reps]

        return cache, loss.detach()

    def forward_backward(
            self,
            model: nn.Module,
            model_inputs,
            cached_gradients: List[Tensor],
            random_states: List[RandContext],
            no_sync_except_last: bool = False
    ):
        """
        Run the second forward and the backward pass to compute gradient for a model.
        :param model: Encoder model.
        :param model_inputs: Chunked input to the encoder model.
        :param cached_gradients: Chunked gradient cache tensor for each input.
        :param random_states: Each input's device random state during the first forward.
        :param no_sync_except_last: If True, under distributed setup, only trigger gradient reduction across processes
        for the last sub-batch's forward-backward pass.
        """
        if no_sync_except_last:
            sync_contexts = [model.no_sync for _ in range(len(model_inputs) - 1)] + [nullcontext]
        else:
            sync_contexts = [nullcontext for _ in range(len(model_inputs))]

        for x, state, gradient, sync_context in zip(model_inputs, random_states, cached_gradients, sync_contexts):
            with sync_context():
                with state:
                    y = self.model_call(model, x)
                reps = self.get_reps(y)

                surrogate = torch.dot(reps.flatten(), gradient.flatten())
                self.backward_fn(surrogate)
    
    def cache_step(
            self,
            *model_inputs,
            no_sync_except_last: bool = False,
            **loss_kwargs
    ) -> Tensor:
        """
        Run a cached step to compute gradient over the inputs.
        :param model_inputs: Input to each encoder model. Should be in similar order as the class's model.
        :param no_sync_except_last: If True, under distributed setup, for each model, only trigger gradient reduction
        across processes for the last sub-batch's forward-backward pass.
        :param loss_kwargs: Additional keyword arguments to the loss function.
        :return: The current's loss.
        """
        all_reps = []
        all_rnd_states = []

        if no_sync_except_last:
            assert all(map(lambda m: isinstance(m, nn.parallel.DistributedDataParallel), self.models)), \
                'Some of models are not wrapped in DistributedDataParallel. Make sure you are running DDP with ' \
                'proper initializations.'

        model_inputs = model_inputs[0]
        model_inputs = [
            {
                'input_ids': model_inputs['input_ids'],
                'attention_mask': model_inputs['attention_mask'],
            },
            {
                'input_ids': model_inputs['doc_input_ids'],
                'attention_mask': model_inputs['doc_attention_mask'],
            }
        ]
        
        if (self.multi_pos > 1):
            for k in ['input_ids', 'attention_mask']:
                model_inputs[1][k] = model_inputs[1][k].flatten(0, 1)

        model_inputs = [self.split_inputs(x, chunk_size) for x, chunk_size in zip(model_inputs, self.chunk_sizes)]
        # print('model_inputs : ', model_inputs)

        for model, x in zip(self.models, model_inputs):
            model_reps, rnd_states = self.forward_no_grad(model, x)
            all_reps.append(model_reps)
            all_rnd_states.append(rnd_states)

        cache, loss = self.build_cache(*all_reps, **loss_kwargs)
        cache = [c.split(chunk_size) for c, chunk_size in zip(cache, self.chunk_sizes)]

        for model, x, model_cache, rnd_states in zip(
                self.models, model_inputs, cache, all_rnd_states):
            self.forward_backward(model, x, model_cache, rnd_states, no_sync_except_last=no_sync_except_last)

        return loss