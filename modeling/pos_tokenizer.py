import nltk
import torch
from transformers import BertTokenizer
from transformers.utils import logging


class POSTokenizer(BertTokenizer):
    # universal pos tags (https://aclanthology.org/L12-1115/)
    # fmt: off
    POS_TAGS = [ "NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "NUM", "CONJ", "PRT", ".", "X", ]
    # fmt: on
    # POS_TAGS needs to have B_ and I_ prefixes
    POS_TAGS = (
        ["<START>", "<STOP>", "<MASK>", "<PAD>"]
        + [f"B_{tag}" for tag in POS_TAGS]
        + [f"I_{tag}" for tag in POS_TAGS]
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eos_token = self.sep_token
        self.pos_tag2idx = {tag: i for i, tag in enumerate(self.POS_TAGS)}
        self.pos_idx2tag = {i: tag for i, tag in enumerate(self.POS_TAGS)}
        self.logger = logging.get_logger("transformers")
        self.logger.warning(f"{__name__}: you can safely ignore the warning above.")

    def pos_tag_tokenizer(self, pos_tags):
        return [self.pos_tag2idx[tag] for tag in pos_tags]

    def __call__(
        self,
        text,
        text_pair=None,
        text_target=None,
        text_pair_target=None,
        **kwargs,
    ):
        def _call_in_pairs(text=None, text_pair=None):
            if isinstance(text, str):
                text = [text]
            if text and not text_pair:
                return self.batch_encode_plus(batch_text_or_text_pairs=text, **kwargs)

            assert type(text) == type(text_pair), "text pairs should be of same type."
            if isinstance(text_pair, str):
                text_pair = [text_pair]
            if type(text) == list:
                assert len(text) == len(text_pair), "text lists should be of same length."
                return self.batch_encode_plus(
                    batch_text_or_text_pairs=list(zip(text, text_pair)), **kwargs
                )
            else:
                raise ValueError(f"Unsupported type: {type(text)}")

        output = _call_in_pairs(text=text, text_pair=text_pair)
        if text_target:
            temp = _call_in_pairs(text=text_target, text_pair=text_pair_target)
            output["labels"] = temp["input_ids"]
            output["label_pos_tag_ids"] = temp["pos_tag_ids"]

        return output

    def make_pos_tags(self, all_sents, add_special_tokens=True):
        all_pos_tags = [nltk.pos_tag(sent, tagset="universal") for sent in all_sents]
        all_pos_tags = [[tag for _, tag in pt] for pt in all_pos_tags]

        tok_pos_tags = []
        for sent, pos_tags in zip(all_sents, all_pos_tags):
            temp = []
            if add_special_tokens:
                temp.append("<START>")
            pos_tag_idx = -1
            """
            Few points to note:
            - .startswith("##") does not work because special symbols and numbers do not contain "##"
              Example: _Contributo_ = ['_', 'con', '##tri', '##bu', '##to', '_']
            - we cannot use self.tokenize(text) because of edge cases like `don't` where 
              BertTokenizer splits it as ["don", "'", "t'], but nltk.word_tokenize splits it 
              as ["do", "n't"]. This means we cannot align the tokens from both the tokenizers.
            """
            for word in sent:
                for wid, _ in enumerate(self.tokenize(word)):
                    if wid == 0:
                        pos_tag_idx += 1
                        temp.append("B_" + pos_tags[pos_tag_idx])
                    else:
                        temp.append("I_" + pos_tags[pos_tag_idx])
            if add_special_tokens:
                temp.append("<STOP>")
            tok_pos_tags.append(temp)

        return [self.pos_tag_tokenizer(tpt) for tpt in tok_pos_tags]

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs,
        padding=False,
        truncation=False,
        return_pos_tag_ids=False,
        return_tensors=None,
        add_special_tokens=True,
        **kwargs,
    ):
        if not return_pos_tag_ids:
            self.logger.warning("return_pos_tag_ids is set to False")
            return super().batch_encode_plus(
                batch_text_or_text_pairs,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors,
                add_special_tokens=add_special_tokens,
                **kwargs,
            )

        single_sent = True
        if isinstance(batch_text_or_text_pairs[0], (list, tuple)):
            single_sent = False

        if single_sent:
            all_sents = [nltk.word_tokenize(t) for t in batch_text_or_text_pairs]
            super_input = [" ".join(w) for w in all_sents]
        else:
            all_sents = [
                (nltk.word_tokenize(t1), nltk.word_tokenize(t2))
                for t1, t2 in batch_text_or_text_pairs
            ]
            super_input = [(" ".join(w1), " ".join(w2)) for w1, w2 in all_sents]

        ret_dict = super().batch_encode_plus(
            super_input,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )

        if single_sent:
            tok_pos_tag_ids = self.make_pos_tags(all_sents, add_special_tokens)
        else:
            all_sents1, all_sents2 = zip(*all_sents)
            tok_pos_tag_ids1 = self.make_pos_tags(all_sents1, add_special_tokens=False)
            tok_pos_tag_ids2 = self.make_pos_tags(all_sents2, add_special_tokens=False)
            tok_pos_tag_ids = [
                [self.pos_tag2idx["<START>"]]
                + t1
                + [self.pos_tag2idx["<STOP>"]]
                + t2
                + [self.pos_tag2idx["<STOP>"]]
                for t1, t2 in zip(tok_pos_tag_ids1, tok_pos_tag_ids2)
            ]

        # if everything is correct, the length of tok_pos_tag_ids will never be lesser than the length of input_ids
        for i, tpt in enumerate(tok_pos_tag_ids):
            if len(tpt) > len(ret_dict["input_ids"][i]):
                if add_special_tokens:
                    tok_pos_tag_ids[i] = tpt[: len(ret_dict["input_ids"][i]) - 1]
                    tok_pos_tag_ids[i].append(self.pos_tag2idx["<STOP>"])
                else:
                    tok_pos_tag_ids[i] = tpt[: len(ret_dict["input_ids"][i])]
        if padding or truncation:
            tok_pos_tag_ids = [
                tpt + [self.pos_tag2idx["<PAD>"]] * (len(ret_dict["input_ids"][j]) - len(tpt))
                for j, tpt in enumerate(tok_pos_tag_ids)
            ]
        if return_tensors == "pt":
            tok_pos_tag_ids = torch.tensor(tok_pos_tag_ids).long()
        elif return_tensors is not None:
            raise ValueError(f"{self.__class__.__name__} supports only 'pt' return_tensors")
        ret_dict["pos_tag_ids"] = tok_pos_tag_ids
        return ret_dict
