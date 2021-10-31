# GlyphBERT


state_dict is a dict:
+ "config": the config of pretraining,
+ "training_state": the training state of pretraining,
+ "model": GlyphBERT's parameters

usage:

```python
from src.GlyphBERT import GlyphBERT
from src.GlyphDataset import GlyphDataset
import torch

# use_res2bert = True
# cnn_and_embed_mat = True

checkpoint = torch.load(config['state_dict'], map_location='cpu')
model = GlyphBERT(config)
model.load_state_dict(checkpoint['model'])

sequence_outputs = model.glyph_bert_sequence_outputs(
    input_ids, token_type_ids, image_input, attention_mask, unique_ids
)
# unique_ids is a list of appeared tokens in input_ids. refer to GlyphDataset.prepare_dataset and collate_fn
# image_input is the tokens' image correspond to the order in unique_ids

```