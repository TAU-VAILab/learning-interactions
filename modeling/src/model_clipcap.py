import pytorch_lightning as pl
import torch
from torch import nn
from transformers import CLIPModel
import numpy as np
from clipcap import ClipCaptionModel

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class HHICCModel(pl.LightningModule):

    def __init__(self, model_cfg, BOS=None, EOS=None, PAD=None, clipcap_pretrained=None, lr=1e-5):

        super().__init__()
        
        self.lr = lr
        self.betas = (model_cfg['beta1'], model_cfg['beta2'])
        self.weight_decay = model_cfg['weight_decay']

        self.n_output_tokens = model_cfg['n_output_tokens']

        self.clip = CLIPModel.from_pretrained(model_cfg['clip_pretrained'])

        self.prefix_length = model_cfg['prefix_length']

        self.cc = ClipCaptionModel(self.prefix_length)
        kwargs = dict()
        if not torch.cuda.is_available():
            kwargs['map_location'] = torch.device('cpu')

        print('Loading CLIPCap pretrained weights:', clipcap_pretrained)
        self.cc.load_state_dict(torch.load(clipcap_pretrained, **kwargs))

        self.freeze_encoder = model_cfg['freeze_encoder']
        if self.freeze_encoder:
            for param in self.clip.parameters():
                param.requires_grad = False
        
        self.BOS = BOS
        self.EOS = EOS
        self.PAD = PAD

        self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=self.PAD)
        # ^overrides default mean reduction to allow sample weights


    def embed_input(self, encoder_inputs):
        # encapsulated in function so that this can be cached for autoregressive decoding

        assert 'pixel_values' in encoder_inputs
        bsz = encoder_inputs['pixel_values'].shape[0]

        V = self.clip.vision_model(pixel_values=encoder_inputs['pixel_values']).pooler_output
        # ^ shape: (b, 1, 768)
        E = self.clip.visual_projection(V)
        # ^ shape: (b, 512)

        # convert to prefix embedding:
        E = self.cc.clip_project(E).reshape(bsz, self.prefix_length, -1)
        # shape: (b, 10, 768)

        output = {
            'embedding': E,
        }
        return output


    def forward(self, encoder_inputs=None, input_ids=None,
                decode=True, E=None,
                **kwargs): # kwargs needed so extra keys don't cause an error

        # encoder_inputs: preprocessed image
        # input_ids: token ids for autoregressive decoding
        # E: cached V&L embedding
        # ^ if passed, skips image encoder

        assert not (encoder_inputs is None and E is None), 'Need non-empty input data'

        if E is None:
            obj = self.embed_input(encoder_inputs)
            E = obj['embedding']
        
        out = {}
        
        if decode:
            assert input_ids is not None, 'Missing input_ids'

            embedding_text = self.cc.gpt.transformer.wte(input_ids)
            # ^ shape: (b, k, 768)
            embedding_cat = torch.cat((E, embedding_text), dim=1)
            # ^ shape: (b, 10+k, 768)

            lm_output = self.cc.gpt(
                inputs_embeds=embedding_cat
            )

            out['lm_logits'] = lm_output.logits[:, self.prefix_length:]
            # ^ shape: (b, k, v)
        
        return out

    def training_step(self, batch, batch_idx):
        bsz = batch['batch_size']
        model_inputs = batch['model_inputs']
        labels = batch['labels']
        sample_weights = batch['weight']
        model_output = self(**model_inputs)
        lm_logits = model_output['lm_logits']

        b = lm_logits.shape[0] # batch size
        n = lm_logits.shape[1] - 1 # number of tokens (when shifted by 1)

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_per_token = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ) # shape (b * n,)
        loss_per_token = loss_per_token.view(b, n)
        # ^ shape (b, n)
        
        weights = sample_weights.unsqueeze(-1) # shape (b, 1) [can be broadcast to (b, n)]
        loss = (loss_per_token * weights).mean() # scalar
        self.log('train', loss, batch_size=bsz)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay)

    @torch.no_grad()
    def run_beam_search(
        self, encoder_inputs=None, E=None,
        beam_k=5, max_length=20, beam_EOS=None, start_tokens=None,
        **kwargs):
        with torch.no_grad():

            if beam_EOS is None:
                beam_EOS = self.EOS
            if start_tokens is None:
                start_tokens = [self.BOS]

            if E is None:
                obj = self.embed_input(encoder_inputs, **kwargs)
                E = obj['embedding']

            bsz = encoder_inputs['pixel_values'].shape[0] if E is None else E.shape[0]
            assert bsz == 1, 'run_beam_search requires batch size 1'

            beams = [(start_tokens, 0)]
            finished_beams = []

            for i in range(max_length):

                if len(beams) == 0:
                    break

                input_ids = torch.tensor([beam[0] for beam in beams]).to(self.device)
                iids = input_ids.shape[0]
                obj = self(input_ids=input_ids, E=E.expand(iids, -1, -1), **kwargs)
                logits = obj['lm_logits'][:, -1, :]
                logits_normed = logits.log_softmax(dim=-1)

                top_logits = logits_normed.topk(beam_k, dim=-1)
                top_vals = top_logits.values
                top_indices = top_logits.indices

                candidates = []
                for j, (beam, score) in enumerate(beams):
                    for k in range(beam_k):
                        new_idx = top_indices[j, k].item()
                        delta_score = top_vals[j, k].item()
                        cand = (
                            beam + [new_idx], score + delta_score
                        )
                        candidates.append(cand)
                candidates = sorted(candidates, key=lambda x: x[1])
                beams = candidates[-beam_k:]

                for (beam, score) in beams:
                    if beam[-1] == beam_EOS:
                        finished_beams.append((beam, score))
                beams = [
                    (beam, score)
                    for beam, score in beams
                    if beam[-1] != beam_EOS
                ]
        if len(finished_beams) == 0:
            # edge case: none of the beams terminated
            # just use the best one without EOS
            finished_beams = [beams[0]]
        finished_beams = sorted(finished_beams, key=lambda x: x[1], reverse=True)[:beam_k]
                
        return finished_beams

    @torch.no_grad()
    def generate_bsz1(self, encoder_inputs=None, E=None, max_length=20,
                    do_beam=False, beam_k=5,
                    **kwargs):

        assert self.BOS is not None and self.EOS is not None, 'BOS and EOS must be defined for generation'

        bsz = encoder_inputs['pixel_values'].shape[0] if E is None else E.shape[0]
        assert bsz == 1, 'generate_bsz1 requires batch size 1'
        with torch.no_grad():
            if E is None: # otherwise, embedding already calculated
                # only calculate V&L embedding once (more efficient than recalculating at each iteration):
                obj = self.embed_input(encoder_inputs, **kwargs)
                E = obj['embedding']

            if do_beam:

                raise NotImplementedError
                # implemented in separate method
            else:
                # greedy decoding

                tokens = [self.BOS]
                
                for i in range(max_length):
                    input_ids = torch.tensor([tokens], device=self.device)
                    obj = self(input_ids=input_ids, E=E, **kwargs)

                    logits = obj['lm_logits'][0, -1, :]
                    idx = logits.argmax().item() # greedy decoding
                    tokens.append(idx)

                    if idx == self.EOS:
                        break
                out = {
                    'tokens': tokens
                }
                return out

    @torch.no_grad()
    def generate(self, encoder_inputs, max_length=20, **kwargs):
        # Note: for now, autoregressive decoding uses batch size 1
        bsz = encoder_inputs['pixel_values'].shape[0]
        outputs = []
        with torch.no_grad():
            for i in range(bsz):
                encoder_inputs_bsz1 = {
                    k: v[i][None]
                    for k, v in encoder_inputs.items()
                }
                obj = self.generate_bsz1(encoder_inputs_bsz1, max_length=max_length, **kwargs)
                outputs.append(obj)
        return outputs # list of dicts