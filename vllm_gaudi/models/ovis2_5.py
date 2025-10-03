import torch
from typing import Union
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.ovis2_5 import (Ovis2_5, Ovis2_5MultiModalProcessor, OvisVideoPatchInputs,
                                                  Ovis2_5ProcessingInfo, Ovis2_5DummyInputsBuilder)
from vllm.model_executor.models.ovis import OvisImagePatchInputs

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm_gaudi.extension.bucketing.vision import HPUVisionBucketManager

@MULTIMODAL_REGISTRY.register_processor(Ovis2_5MultiModalProcessor,
                                        info=Ovis2_5ProcessingInfo,
                                        dummy_inputs=Ovis2_5DummyInputsBuilder)
class HpuOvis2_5(Ovis2_5):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def _process_image_input(
        self, image_input: Union[OvisImagePatchInputs, OvisVideoPatchInputs]
    ) -> MultiModalEmbeddings:
        image_patches_flat = image_input["flat_data"]
        indicator_tokens = image_input["indicator_tokens"]
        grid_thws = image_input["grids"]

        target_dtype = self.visual_tokenizer.dtype

        visual_embeds, grid_thws = self.vision_bucket_manager.pad_multimodal_data(
            image_patches_flat.to(target_dtype), grid_thws)

        visual_tokens = self.visual_tokenizer(visual_embeds, grid_thws)
        visual_embeds = self.vte(visual_tokens)  # 1:1 numeric eq.
        indicator_embeds = self.vte(indicator_tokens)
        padded_patches_per_image = [
            grid[1] * grid[2] // (self.config.vit_config.hidden_stride**2)
            for grid in grid_thws
        ]

        visual_embeds_per_image = visual_embeds.split(padded_patches_per_image,
                                                      dim=0)
        indicator_per_image = list(
            map(lambda x: 2 if x > 1 else x + 2, padded_patches_per_image))
        indicator_embeds_per_image = indicator_embeds.split(
            indicator_per_image)

        vision_embeddings = []
        for idx, (indicator, visual) in enumerate(
                zip(indicator_embeds_per_image, visual_embeds_per_image)):
            vision_embeddings_per_image = []
            visual = visual.unsqueeze(0)
            for i in range(visual.shape[0]):
                vision_embeddings_per_image.append(
                    torch.cat([indicator[i:i + 1], visual[i]], dim=0))
            vision_embeddings_per_image.append(indicator[i + 1:])
            vision_embeddings.append(
                torch.cat(vision_embeddings_per_image, dim=0))
        return tuple(vision_embeddings)

