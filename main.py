import os
import torch
import fairseq

print(torch.backends.quantized.supported_engines)
torch.backends.quantized.engine = 'qnnpack'
# model_path='wav2vec_small_960h.pt'
# path, checkpoint = os.path.split(model_path)

# # overrides with audio_finetuning task
# overrides = {
#     "task": 'audio_finetuning',
#     "data": path,
# }
# models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
#     fairseq.utils.split_paths(checkpoint, separator="\\"),
#     arg_overrides=overrides,
#     strict=True,
# )
# model = models[0]


# scripted_module = torch.jit.script(model)
# Export full jit version model (not compatible mobile interpreter), leave it here for comparison
# scripted_module.save("deeplabv3_scripted.pt")
# Export mobile interpreter version model (compatible with mobile interpreter)
# optimized_scripted_module = optimize_for_mobile(scripted_module)
# optimized_scripted_module._save_for_lite_interpreter("deeplabv3_scripted.ptl")