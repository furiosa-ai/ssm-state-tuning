


from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


class MambaLMHeadModelPeft(MambaLMHeadModel):
    def __init__(self, config: MambaConfig, initializer_cfg=None, device=None, dtype=None, **kwargs) -> None:
        super().__init__(config, initializer_cfg, device, dtype, **kwargs)

    def get_mamba_blocks(self):
        blocks = self.backbone.layers
        mamba_blocks = [b.mixer for b in blocks]
        return mamba_blocks
    
    def split_layers(self):
        for m in self.get_mamba_blocks():
            m.split_layers()

    def combine_layers(self):
        for m in self.get_mamba_blocks():
            m.combine_layers()

    @property
    def word_embeddings(self):
        return self.backbone.embedding

    @property
    def device(self):
        return self.backbone.embedding.weight.device
