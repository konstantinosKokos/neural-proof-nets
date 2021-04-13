from typing import NoReturn

from torch.nn import Dropout, Sequential, LayerNorm
from torch.utils.data.dataloader import DataLoader

from ..neural.sinkhorn import sinkhorn_fn_no_exp as sinkhorn
from ..neural.utils import *
from ..neural.schedules import Scheduler
from ..neural.embedding import ComplexEmbedding
from ..parsing.postprocessing import TypeParser, Analysis
from ..neural.transformer import make_encoder, make_decoder, FFN

from transformers import RobertaModel

from tqdm import tqdm


class Parser(Module):
    def __init__(self, atokenizer: AtomTokenizer, tokenizer: Tokenizer, dec_dim: int = 64, device: str = 'cpu'):
        super(Parser, self).__init__()
        self.enc_dim = 768
        self.dec_dim = dec_dim
        self.num_embeddings = len(atokenizer)
        self.device = device
        self.atom_tokenizer = atokenizer
        self.type_parser = TypeParser()
        self.tokenizer = tokenizer
        self.dropout = Dropout(0.1)
        self.enc_heads = 8
        self.dec_heads = 8
        self.d_atn_dec = self.dec_dim//self.dec_heads

        self.word_encoder = RobertaModel.from_pretrained("pdelobelle/robbert-v2-dutch-base").to(device)
        self.supertagger = make_decoder(num_layers=6, num_heads_enc=self.enc_heads, num_heads_dec=self.dec_heads,
                                        d_encoder=self.enc_dim, d_decoder=self.dec_dim,
                                        d_atn_enc=self.enc_dim//self.enc_heads, d_atn_dec=self.d_atn_dec,
                                        d_v_enc=self.enc_dim//self.enc_heads, d_v_dec=self.dec_dim//self.dec_heads,
                                        d_interm=self.dec_dim * 2, dropout_rate=0.1).to(device)
        self.atom_embedder = ComplexEmbedding(self.num_embeddings, self.dec_dim//2).to(device)
        self.linker = make_encoder(num_layers=3, num_heads=self.enc_heads, d_intermediate=self.dec_dim * 4,
                                   dropout=0.15, d_model=self.dec_dim * 2, d_k=(self.dec_dim * 2)//self.dec_heads,
                                   d_v=(self.dec_dim * 2)//self.dec_heads).to(device)
        self.pos_transformation = Sequential(
            FFN(self.dec_dim * 2, self.dec_dim, 0.1, self.dec_dim//2), LayerNorm(self.dec_dim//2, eps=1e-12)
        ).to(device)
        self.neg_transformation = Sequential(
            FFN(self.dec_dim * 2, self.dec_dim, 0.1, self.dec_dim // 2), LayerNorm(self.dec_dim//2, eps=1e-12)
        ).to(device)

    def forward(self, *args) -> NoReturn:
        raise NotImplementedError('Forward not implemented.')

    @staticmethod
    def sinkhorn(x: Tensor, iters: int, tau: int = 1) -> Tensor:
        return sinkhorn(x, tau=tau, iters=iters)

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = torch.FloatTensor) -> Module:
        self.device = self.device if device is None else device
        return super(Parser, self).to(device=device, dtype=dtype)

    @torch.no_grad()
    def make_mask(self, inps: Tensor, padding_id: int) -> Tensor:
        mask = torch.ones_like(inps)
        mask[inps.eq(padding_id)] = 0
        return mask.to(self.device)

    def make_word_mask(self, lexical_ids: Tensor) -> Tensor:
        return self.make_mask(lexical_ids, self.tokenizer.core.pad_token_id)

    def make_atom_mask(self, atom_ids: Tensor) -> Tensor:
        return self.make_mask(atom_ids, self.atom_tokenizer.pad_token_id).unsqueeze(1).repeat(1, atom_ids.shape[1], 1)

    @torch.no_grad()
    def make_atom_mask_from_lens(self, lens: list[int]) -> Tensor:
        ones = torch.ones((len(lens), max(lens), max(lens)))
        for i, l in enumerate(lens):
            ones[i, :, l:] = 0
        return ones.to(self.device)

    @torch.no_grad()
    def make_decoder_mask(self, b: int, n: int) -> Tensor:
        upper_triangular = torch.triu(torch.ones(b, n, n), diagonal=1)
        return (torch.ones(b, n, n) - upper_triangular).to(self.device)

    @torch.no_grad()
    def extend_mask(self, mask: Tensor, size: int) -> Tensor:
        return mask.unsqueeze(1).repeat(1, size, 1)

    def count_sep(self, xs: Tensor) -> Tensor:
        return count_sep(xs, self.atom_tokenizer.sep_token_id)

    def precode(self, lexical_token_ids: Tensor, symbol_ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        b, s_in = lexical_token_ids.shape
        s_out = symbol_ids.shape[1]

        encoder_mask = self.make_word_mask(lexical_token_ids)
        encoder_output = self.encode_words(lexical_token_ids, encoder_mask)

        decoder_mask = self.make_decoder_mask(b=b, n=s_out)
        atom_embeddings = self.atom_embedder(symbol_ids.to(self.device))

        extended_encoder_mask = self.extend_mask(encoder_mask, s_out).to(self.device)

        return encoder_output, atom_embeddings, decoder_mask, extended_encoder_mask

    def predict_atoms(self, reprs: Tensor, t: int) -> Tensor:
        return self.atom_embedder.invert(reprs, t)

    def encode_words(self, lexical_token_ids: Tensor, encoder_mask: Tensor) -> Tensor:
        out = self.word_encoder(lexical_token_ids.to(self.device), attention_mask=encoder_mask.to(self.device))
        return self.dropout(out['last_hidden_state'])

    def encode_atoms(self, atom_reprs: Tensor, atom_mask: Tensor, decoder_reprs: Tensor) -> Tensor:
        s_out = atom_reprs.shape[1]
        if s_out == 0:
            return atom_reprs
        return self.linker((self.dropout(torch.cat((atom_reprs, decoder_reprs), dim=-1)), atom_mask))[0]

    def link(self, atom_reprs: Tensor, atom_mask: Tensor, decoder_reprs: Tensor, pos_idxes: list[list[Tensor]],
             neg_idxes: list[list[Tensor]], exclude_singular: bool = True, sinkhorn_iters: int = 3) -> list[Tensor]:

        atom_reprs = self.encode_atoms(atom_reprs, atom_mask, decoder_reprs)

        _positives: list[list[Tensor]] = make_sinkhorn_inputs(atom_reprs, pos_idxes, self.device)
        _negatives: list[list[Tensor]] = make_sinkhorn_inputs(atom_reprs, neg_idxes, self.device)

        positives = [tensor for tensor in chain.from_iterable(_positives) if min(tensor.size()) != 0]
        negatives = [tensor for tensor in chain.from_iterable(_negatives) if min(tensor.size()) != 0]

        distinct_shapes = {tensor.size()[0] for tensor in positives}
        if exclude_singular:
            distinct_shapes = distinct_shapes.difference({1})
        distinct_shapes = sorted(distinct_shapes)

        matches: list[Tensor] = []

        all_shape_positives: list[Tensor] \
            = [self.pos_transformation(self.dropout(torch.stack([tensor for tensor in positives
                                                                 if tensor.size()[0] == shape])))
               for shape in distinct_shapes]
        all_shape_negatives: list[Tensor] \
            = [self.neg_transformation(self.dropout(torch.stack([tensor for tensor in negatives
                                                                 if tensor.size()[0] == shape])))
               for shape in distinct_shapes]

        for this_shape_positives, this_shape_negatives in zip(all_shape_positives, all_shape_negatives):
            weights = torch.bmm(this_shape_positives,
                                this_shape_negatives.transpose(2, 1))
            matches.append(self.sinkhorn(weights, iters=sinkhorn_iters))
        return matches

    def link_train(self, *args, **kwargs) -> list[Tensor]:
        return self.link(*args, **kwargs, sinkhorn_iters=3)

    def link_eval(self, *args, **kwargs) -> list[Tensor]:
        return self.link(*args, **kwargs, sinkhorn_iters=3)

    def link_slow(self, atom_reprs: Tensor, atom_mask: Tensor, decoder_reprs: Tensor, pos_idxes: list[list[Tensor]],
                  neg_idxes: list[list[Tensor]], sinkhorn_iters: int = 5) \
            -> tuple[list[list[Tensor]], list[list[Tensor]]]:

        atom_reprs = self.encode_atoms(atom_reprs, atom_mask, decoder_reprs)

        _positives: list[list[Tensor]] = make_sinkhorn_inputs(atom_reprs, pos_idxes, self.device)
        _negatives: list[list[Tensor]] = make_sinkhorn_inputs(atom_reprs, neg_idxes, self.device)

        ws = []
        ret = []
        for sent in zip(_positives, _negatives):
            sent = list(zip(*sent))
            local_ws = []
            local_rs = []
            for pos, neg in sent:
                weights = self.pos_transformation(self.dropout(pos)) @ \
                          self.neg_transformation(self.dropout(neg)).transpose(-1, -2)
                local_ws.append(weights)
                local_rs.append(self.sinkhorn(weights.unsqueeze(0), iters=sinkhorn_iters))
            ws.append(local_ws)
            ret.append(local_rs)
        return ws, ret

    def encode_with_decoder(self, symbol_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, s_out = symbol_ids.shape
        decoder_mask = self.make_decoder_mask(b=b, n=s_out).to(self.device)
        atom_embeddings = self.atom_embedder.embed(symbol_ids.to(self.device), 0)
        return self.supertagger((atom_embeddings, decoder_mask))[0], atom_embeddings, decoder_mask

    def decode_train(self, lexical_token_ids: Tensor, symbol_ids: Tensor) \
            -> tuple[Tensor, Tensor, Tensor, Tensor]:

        tmp = self.precode(lexical_token_ids, symbol_ids)
        encoder_output, atom_embeddings, decoder_mask, extended_encoder_mask = tmp

        output_reprs = self.supertagger((encoder_output, extended_encoder_mask, atom_embeddings, decoder_mask))[2]

        return output_reprs, encoder_output, atom_embeddings, extended_encoder_mask

    def decode_greedy(self, lexical_token_ids: Tensor, max_decode_length: Optional[int] = None,
                      length_factor: int = 5) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        b, s_in = lexical_token_ids.shape

        if max_decode_length is None:
            s_out = length_factor * s_in
        else:
            s_out = max_decode_length

        encoder_mask = self.make_word_mask(lexical_token_ids)
        encoder_output = self.encode_words(lexical_token_ids, encoder_mask)

        extended_encoder_mask = self.extend_mask(encoder_mask, s_out).to(self.device)
        decoder_mask = self.make_decoder_mask(b, s_out)

        output_symbols = (torch.ones(b) * self.atom_tokenizer.sos_token_id).long().to(self.device)
        decoder_input = (self.atom_embedder.embed(output_symbols, 0)).unsqueeze(1)
        output_symbols = output_symbols.unsqueeze(1)
        decoder_output = torch.empty(b, s_out, self.dec_dim).to(self.device)

        for t in range(s_out):
            _decoder_tuple_input = (encoder_output, extended_encoder_mask,
                                    decoder_input, decoder_mask[:, :t + 1, :t + 1])
            repr_t = self.supertagger(_decoder_tuple_input)[2][:, -1]
            prob_t = self.predict_atoms(repr_t, t+1)
            class_t = prob_t.argmax(dim=-1)
            output_symbols = torch.cat([output_symbols, class_t.unsqueeze(1)], dim=1)
            next_embedding = self.atom_embedder.embed(class_t, t+1).unsqueeze(1)
            decoder_input = torch.cat([decoder_input, next_embedding], dim=1)
            decoder_output[:, t] = repr_t
        return output_symbols, decoder_input, encoder_output, extended_encoder_mask, decoder_output

    @torch.no_grad()
    def decode_beam(self, lexical_token_ids: Tensor, beam_width: int, stop_at: list[int],
                    max_decode_length: Optional[int] = None, length_factor: int = 5, alpha: float = 0.) \
            -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

        def backward_index(idx: int) -> tuple[int, int]:
            return idx // beam_width, idx - (idx // beam_width) * beam_width

        def lp(beam_len: int) -> float:
            return (beam_len + 5)**alpha/6**alpha

        def invariance_check(xs: Tensor) -> bool:
            return (xs.bincount()[5:36] % 2).bool().logical_not().all()

        def get_last_span(xs: Tensor) -> list[str]:
            xs = xs.tolist()
            try:
                start = len(xs) - 1 - xs[::-1].index(self.atom_tokenizer.sep_token_id)
            except ValueError:
                start = 0
            return [self.atom_tokenizer.inverse_atom_map[x] for x in xs[start:]]

        batch_size, s_in = lexical_token_ids.shape
        stop_at = torch.tensor(stop_at, dtype=torch.long, device=self.device).unsqueeze(1)
        s_out = length_factor * s_in if max_decode_length is None else max_decode_length
        encoder_mask = self.make_word_mask(lexical_token_ids)
        encoder_output = self.encode_words(lexical_token_ids, encoder_mask)
        extended_encoder_mask = encoder_mask.unsqueeze(1).repeat(1, s_out, 1).to(self.device)
        decoder_mask = self.make_decoder_mask(batch_size, s_out)

        sos_tokens = (torch.ones(batch_size, device=self.device) * self.atom_tokenizer.sos_token_id).long()
        decoder_input = self.atom_embedder(sos_tokens).view(batch_size, 1, 1, self.dec_dim).repeat(1, beam_width, 1, 1)
        sos_tokens = sos_tokens.unsqueeze(1)
        decoder_output = torch.zeros(batch_size, beam_width, s_out, self.dec_dim, device=self.device)
        output_symbols = torch.zeros(batch_size, beam_width, s_out, dtype=torch.long, device=self.device)
        beam_scores = torch.zeros(batch_size, beam_width, device=self.device)

        masked_probs = torch.ones(1, 1, len(self.atom_tokenizer), device=self.device) * -1e10
        masked_probs[:, :, self.atom_tokenizer.pad_token_id] = 0

        for t in range(s_out):
            # the next repr of each batch-beam combination
            repr_t = torch.empty(batch_size, beam_width, self.dec_dim, device=self.device)
            for beam in range(beam_width):
                _decoder_tuple_input = (encoder_output, extended_encoder_mask,
                                        decoder_input[:, beam], decoder_mask[:, :t + 1, :t + 1])
                repr_t[:, beam] = self.supertagger(_decoder_tuple_input)[2][:, -1]
            logprobs_t = self.predict_atoms(repr_t, t+1).log_softmax(dim=-1)                        # B, K, S

            open_beams = self.count_sep(output_symbols).lt(stop_at).unsqueeze(-1)                   # B, K, (1)
            if not open_beams.flatten().any():
                break

            # only trust first beam at t0
            if t == 0:
                logprobs_t[:, 1:] = -1e10

            logprobs_t = torch.where(open_beams, logprobs_t, masked_probs)

            # local_scores[k1] contains the k2 best branches originating from beam[k1]
            local_scores, local_steps = list(zip(*[logprobs_t[:, k].topk(k=beam_width, dim=-1)
                                                   for k in range(beam_width)]))

            local_scores = torch.stack(local_scores, dim=1) + beam_scores.unsqueeze(-1)                 # B, K1, K2
            local_steps = torch.stack(local_steps, dim=1)                                               # B, K1, K2

            best_scores, best_sources = local_scores.view(batch_size, -1).topk(dim=-1, k=beam_width)
            best_source_idxes = [[backward_index(idx) for idx in best_source] for best_source in best_sources.tolist()]

            new_decoder_output = torch.zeros(batch_size, beam_width, s_out, self.dec_dim, device=self.device)
            new_decoder_input = torch.zeros(batch_size, beam_width, t + 1, self.dec_dim, device=self.device)
            new_output_symbols = torch.zeros(batch_size, beam_width, s_out, dtype=torch.long, device=self.device)

            for sent in range(batch_size):
                for beam in range(beam_width):
                    origin, target = best_source_idxes[sent][beam]
                    new_decoder_output[sent, beam, :t] = decoder_output[sent, origin, :t]
                    new_decoder_output[sent, beam, t] = repr_t[sent, origin]
                    new_decoder_input[sent, beam, :t + 1] = decoder_input[sent, origin, :t + 1]
                    new_output_symbols[sent, beam, :t] = output_symbols[sent, origin, :t]
                    new_output_symbols[sent, beam, t] = local_steps[sent, origin, target]
                    beam_scores[sent, beam] = best_scores[sent, beam]
                    if new_output_symbols[sent, beam, t] == self.atom_tokenizer.sep_token_id:
                        # if not TypeParser.polish_to_type(get_last_span(new_output_symbols[sent, beam])):
                        #     beam_scores[sent, beam] = -1e10
                        if self.count_sep(new_output_symbols[sent, beam, :t + 1]) == stop_at[sent]:
                        #     if not invariance_check(new_output_symbols[sent, beam, :t+1]):
                        #         beam_scores[sent, beam] = -1e10
                        #     else:
                            beam_scores[sent, beam] /= lp(t+1)
            decoder_output = new_decoder_output
            decoder_input = new_decoder_input
            output_symbols = new_output_symbols

            class_t = output_symbols[:, :, t].view(batch_size * beam_width)

            next_embedding = self.atom_embedder.embed(class_t, t + 1).view(batch_size, beam_width, self.dec_dim)

            if t != s_out - 1:
                next_embedding = next_embedding.unsqueeze(2)
                decoder_input = torch.cat([decoder_input, next_embedding], dim=2)
        output_symbols = torch.cat([sos_tokens.unsqueeze(1).repeat(1, beam_width, 1), output_symbols], dim=2)
        return (output_symbols, decoder_input, encoder_output, extended_encoder_mask,
                decoder_output.view(batch_size, beam_width, -1, self.dec_dim), beam_scores)

    def pretrain_decoder_batch(self, batch: Batch, st_loss_fn: Module, l_loss_fn: Module,
                               optimizer: Scheduler, linking_weight: float) -> tuple[float, float]:
        self.train()
        _, types, pos_idxes, neg_idxes, grouped_permutors = batch
        output_reprs, atom_embeddings, atom_mask = self.encode_with_decoder(types)
        type_predictions = self.predict_atoms(output_reprs[:, :-1], 1)
        tagging_loss = st_loss_fn(type_predictions, types[:, 1:].to(self.device))
        link_weights = self.link_train(atom_embeddings, atom_mask, output_reprs, pos_idxes, neg_idxes)
        grouped_permutors = [perm.to(self.device) for perm in grouped_permutors]
        link_loss = l_loss_fn(link_weights, grouped_permutors)
        mutual_loss = link_loss * linking_weight + (1 - linking_weight) * tagging_loss
        mutual_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return tagging_loss.item(), link_loss.item()

    def train_batch(self, batch: Batch, st_loss_fn: Module, l_loss_fn: Module, optimizer: Scheduler,
                    linking_weight: float = 0.5) -> tuple[float, float]:
        self.train()

        words, types, pos_idxes, neg_idxes, grouped_permutors = batch
        atom_mask = self.make_atom_mask(types)

        output_reprs, _, atom_embeddings, _ = self.decode_train(words, types)

        # supertagging
        type_predictions = self.predict_atoms(output_reprs[:, :-1], 1)
        tagging_loss = st_loss_fn(type_predictions, types[:, 1:].to(self.device))

        if linking_weight == 0:
            tagging_loss.backward()
            link_loss = 0
        else:
            # axiom linking
            link_weights = self.link_train(atom_embeddings, atom_mask, output_reprs, pos_idxes, neg_idxes)
            grouped_permutors = [perm.to(self.device) for perm in grouped_permutors]
            link_loss = l_loss_fn(link_weights, grouped_permutors)
            mutual_loss = link_loss * linking_weight + (1 - linking_weight) * tagging_loss
            mutual_loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        return tagging_loss.item(), (link_loss.item() if link_loss != 0 else 0)

    @torch.no_grad()
    def eval_batch(self, batch: Batch, link: bool = True) \
            -> tuple[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]:
        self.eval()

        words, types, pos_idxes, neg_idxes, grouped_permutors = batch
        atom_mask = self.make_atom_mask(types)

        max_length = types.shape[1]
        temp = self.decode_greedy(words, max_decode_length=max_length)
        true_embeddings = self.atom_embedder(types.to(self.device))
        type_predictions, _, _, _, decoder_output = temp
        type_predictions = type_predictions[:, 1:-1]

        # supertagging
        types = types[:, 1:].to(self.device)

        if link:
            # linking
            links = self.link_eval(true_embeddings, atom_mask, decoder_output, pos_idxes, neg_idxes, False)
            permutors = [perm.to(self.device) for perm in grouped_permutors]

            return (measure_linking_accuracy(links, permutors),
                    measure_supertagging_accuracy(type_predictions, types))
        return (0, 1), measure_supertagging_accuracy(type_predictions, types)

    @torch.no_grad()
    def preval_batch(self, batch: Batch) -> tuple[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]:
        self.eval()

        _, types, pos_idxes, neg_idxes, grouped_permutors = batch
        output_reprs, atom_embeddings, atom_mask = self.encode_with_decoder(types)
        type_predictions = self.predict_atoms(output_reprs[:, :-1], t=1).argmax(dim=-1)
        links = self.link_eval(atom_embeddings, atom_mask, output_reprs, pos_idxes, neg_idxes, False)
        permutors = [perm.to(self.device) for perm in grouped_permutors]
        types = types[:, 1:].to(self.device)
        return measure_linking_accuracy(links, permutors), measure_supertagging_accuracy(type_predictions, types)

    def train_epoch(self, dataloader: DataLoader, st_loss_fn: Module, l_loss_fn: Module, optimizer: Scheduler,
                    linking_weight: float = 0.5) -> tuple[float, float]:

        total_l1, total_l2 = 0, 0
        for batch in tqdm(dataloader):
            l1, l2 = self.train_batch(batch, st_loss_fn, l_loss_fn, optimizer, linking_weight=linking_weight)
            total_l1 += l1
            total_l2 += l2
        return total_l1 / len(dataloader), total_l2 / len(dataloader)

    def pretrain_decoder_epoch(self, dataloader: DataLoader, st_loss_fn: Module, l_loss_fn: Module,
                               optimizer: Scheduler, linking_weight: float) -> tuple[float, float]:
        tag_loss, link_loss = 0., 0.
        for batch in tqdm(dataloader):
            tl, ll = self.pretrain_decoder_batch(batch, st_loss_fn, l_loss_fn, optimizer, linking_weight)
            tag_loss += tl
            link_loss += ll
        return tag_loss / len(dataloader), link_loss / len(dataloader)

    def eval_epoch(self, dataloader: DataLoader, link: bool = True) -> tuple[float, float, float]:
        l_total, l_correct, s_total, s_correct, w_total, w_correct = (0.1,) * 6

        for batch in tqdm(dataloader):
            batch_output = self.eval_batch(batch, link)
            (bl_correct, bl_total), ((bs_correct, bs_total), (bw_correct, bw_total)) = batch_output
            s_total += bs_total
            s_correct += bs_correct
            w_total += bw_total
            w_correct += bw_correct
            l_total += bl_total
            l_correct += bl_correct
        return s_correct / s_total, w_correct / w_total, l_correct / l_total

    def preval_epoch(self, dataloader: DataLoader) -> tuple[float, float, float]:
        l_total, l_correct, s_total, s_correct, w_total, w_correct = (0.1,) * 6
        for batch in tqdm(dataloader):
            batch_output = self.preval_batch(batch)
            (bl_correct, bl_total), ((bs_correct, bs_total), (bw_correct, bw_total)) = batch_output
            s_total += bs_total
            s_correct += bs_correct
            w_total += bw_total
            w_correct += bw_correct
            l_total += bl_total
            l_correct += bl_correct
        return s_correct / s_total, w_correct / w_total, l_correct / l_total

    @torch.no_grad()
    def infer(self, sents: list[str], beam_size: int, **kwargs) -> list[list[Analysis]]:
        self.eval()

        sent_lens = [len(sent.split()) + 1 for sent in sents]
        lexical_token_ids = sents_to_batch(sents, self.tokenizer)

        if beam_size == 1:
            type_preds, atom_embeddings, encoder_output, wmask_, decoder_output = self.decode_greedy(
                lexical_token_ids.to(self.device), **kwargs)
            type_preds = [[x] for x in type_preds.tolist()]
            atom_embeddings = atom_embeddings.unsqueeze(1)
            decoder_output = decoder_output.unsqueeze(1)
        else:
            type_preds, atom_embeddings, encoder_output, wmask_, decoder_output, _ = self.decode_beam(
                lexical_token_ids.to(self.device), beam_width=beam_size, stop_at=sent_lens, **kwargs)
            type_preds = type_preds.tolist()

        atom_seqs: list[list[Optional[list[list[str]]]]]
        # filter decoded atoms that count at least as many types as words
        atom_seqs = self.atom_tokenizer.convert_beam_ids_to_polish(type_preds, sent_lens)

        analyses = self.type_parser.parse_beam_batch(sents, atom_seqs)
        valid_for_linking = [((s, b), len(analyses[s][b].polish), analyses[s][b])
                             for s in range(len(analyses))
                             for b in range(len(analyses[s]))
                             if analyses[s][b].valid()]
        if not valid_for_linking:
            return analyses

        ids, atom_lens, valid_analyses = list(zip(*valid_for_linking))
        atom_mask = self.make_atom_mask_from_lens(atom_lens)

        atom_reprs = torch.zeros(len(ids), max(atom_lens), self.dec_dim, device=self.device)
        d_out = torch.zeros(len(ids), max(atom_lens), self.dec_dim, device=self.device)

        for i, (s, b) in enumerate(ids):
            atom_reprs[i] = atom_embeddings[s, b, :max(atom_lens)]
            d_out[i] = decoder_output[s, b, :max(atom_lens)]

        positive_ids, negative_ids = Analysis.to_indices(valid_analyses)
        weights_, links_ = self.link_slow(atom_reprs, atom_mask, d_out, positive_ids, negative_ids)

        weights = [[w.tolist()[0] for w in sent] for sent in weights_]
        links = [[link_weights.argmax(dim=-1).tolist()[0] for link_weights in sent] for sent in links_]
        for va, weight, link in zip(valid_analyses, weights, links):
            va.link_weights = weight
            va.fill_matches(link)

        return analyses

    @torch.no_grad()
    def parse_with_oracle(self, samples: list[Sample]) -> list[list[Analysis]]:
        self.eval()

        sent_lens = list(map(lambda s: len(s.words) + 1, samples))
        words, types, pos_ids, neg_ids = self.atom_tokenizer.samples_to_batch(samples, self.tokenizer)

        tmp = self.decode_train(words, types)
        decoder_output, encoder_output, atom_embeddings, wmask_ = tmp

        type_preds = [[x] for x in types.tolist()]
        atom_embeddings = atom_embeddings.unsqueeze(1)
        # filter decoded atoms that count at least as many types as words
        atom_seqs = self.atom_tokenizer.convert_beam_ids_to_polish(type_preds, sent_lens)

        analyses = self.type_parser.parse_beam_batch([' '.join(s.words) for s in samples], atom_seqs)
        valid_for_linking = [((s, b), len(analyses[s][b].polish), analyses[s][b])
                             for s in range(len(analyses))
                             for b in range(len(analyses[s]))
                             if analyses[s][b].valid()]
        if not valid_for_linking:
            return analyses

        ids, atom_lens, valid_analyses = list(zip(*valid_for_linking))
        atom_mask = self.make_atom_mask_from_lens(atom_lens)

        atom_reprs = torch.zeros(len(ids), max(atom_lens), self.dec_dim, device=self.device)
        word_reprs = torch.zeros(len(ids), encoder_output.shape[1], self.enc_dim, device=self.device)
        wmask = torch.zeros(len(ids), wmask_.shape[1], wmask_.shape[2], device=self.device, dtype=torch.long)
        for i, (s, b) in enumerate(ids):
            atom_reprs[i] = atom_embeddings[s, b, :max(atom_lens)]
            word_reprs[i] = encoder_output[s]
            wmask[i] = wmask_[s]

        positive_ids, negative_ids = Analysis.to_indices(valid_analyses)

        links_ = self.link_slow(atom_reprs, atom_mask, decoder_output, positive_ids, negative_ids)

        links = [[link_weights.argmax(dim=-1).tolist()[0] for _, link_weights in sent] for sent in links_]
        for va, link in zip(valid_analyses, links):
            va.fill_matches(link)
        return analyses
