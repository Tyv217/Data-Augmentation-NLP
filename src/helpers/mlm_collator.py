import torch
from transformers import DataCollatorForWholeWordMask

class MLMCollator(DataCollatorForWholeWordMask):
    
    def collate_fn(self, inputs):
        device = inputs[0]['input_id'].device
        input_ids = torch.stack([input['input_id'] for input in inputs])
        attention_masks = torch.stack([input['attention_mask'] for input in inputs])

        batch = {
            'input_id': input_ids.to(device),
            'attention_mask': attention_masks.to(device),
        }

        if self.mlm_probability is not None:
            input_ids = [input_id for input_id in batch['input_id']]
            masked_inputs = self.torch_call(input_ids)
            batch['input_id'] = masked_inputs["input_ids"].long().to(device)
            batch['label'] = masked_inputs["labels"].long().to(device)

        return batch