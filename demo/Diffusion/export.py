import os
import onnx
import torch
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel
from cuda import cudart
from polygraphy.backend.trt import CreateConfig, Profile
from polygraphy.backend.trt import engine_from_network, network_from_onnx_path, save_engine
import tensorrt as trt

from models import Optimizer


##############
# meta
##############
BATCH_SIZE = 1
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
TEXT_MAX_LENGHT = 77
TEXT_EMBEDDING_SIZE = 768
DEVICE = "cuda"
ONNX_OPT_SET = 16
ONNX_DIR = "anything/onnx"
ENGINE_DIR = "anything/engine"
_HF_CLIP_MODEL_NAME = "/yanbc/workspace/codes/img2img/src_models/anything/text_encoder/"
_HF_UNET_MODEL_NAME = "/yanbc/workspace/codes/img2img/src_models/anything/unet/"
_HF_VAE_MODEL_NAME = "/yanbc/workspace/codes/img2img/src_models/anything/vae/"
_ENABLE_TRT_PREVIEW = False


def build_engine(onnx_path, engine_path, fp16=True, input_profile=None, enable_preview=_ENABLE_TRT_PREVIEW):
    print(f"Building TensorRT engine for {onnx_path}: {engine_path}")
    p = Profile()
    if input_profile:
        for name, dims in input_profile.items():
            assert len(dims) == 3
            p.add(name, min=dims[0], opt=dims[1], max=dims[2])

    preview_features = []
    if enable_preview:
        trt_version = [int(i) for i in trt.__version__.split(".")]
        # FASTER_DYNAMIC_SHAPES_0805 should only be used for TRT 8.5.1 or above.
        if trt_version[0] > 8 or \
            (trt_version[0] == 8 and (trt_version[1] > 5 or (trt_version[1] == 5 and trt_version[2] >= 1))):
            preview_features = [trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805]

    engine = engine_from_network(network_from_onnx_path(onnx_path), config=CreateConfig(fp16=fp16, profiles=[p],
        preview_features=preview_features))
    save_engine(engine, path=engine_path)


##############
# clip
##############
def optimize_clip(onnx_graph, minimal_optimization=False):
    enable_optimization = not minimal_optimization

    # Remove Cast Node to optimize Attention block
    bRemoveCastNode = enable_optimization
    # Insert LayerNormalization Plugin
    bLayerNormPlugin = enable_optimization

    opt = Optimizer(onnx_graph, verbose=False)
    opt.info('CLIP: original')
    opt.select_outputs([0]) # delete graph output#1
    opt.cleanup()
    opt.info('CLIP: remove output[1]')
    opt.fold_constants()
    opt.info('CLIP: fold constants')
    opt.infer_shapes()
    opt.info('CLIP: shape inference')

    if bRemoveCastNode:
        num_casts_removed = opt.remove_casts()
        opt.info('CLIP: removed '+str(num_casts_removed)+' casts')

    if bLayerNormPlugin:
        num_layernorm_inserted = opt.insert_layernorm_plugin()
        opt.info('CLIP: inserted '+str(num_layernorm_inserted)+' LayerNorm plugins')

    opt.select_outputs([0], names=['text_embeddings']) # rename network output
    opt_onnx_graph = opt.cleanup(return_onnx=True)
    opt.info('CLIP: final')
    return opt_onnx_graph


def export_clip_onnx():
    onnx_path = os.path.join(ONNX_DIR, "clip.onnx")
    onnx_opt_path = os.path.join(ONNX_DIR, "clip.opt.onnx")
    torch_model = CLIPTextModel.from_pretrained(_HF_CLIP_MODEL_NAME).to(DEVICE)
    with torch.inference_mode(), torch.autocast(DEVICE):
        inputs = torch.zeros(BATCH_SIZE, TEXT_MAX_LENGHT, dtype=torch.int32, device=DEVICE)
        torch.onnx.export(
                torch_model,
                inputs,
                onnx_path,
                export_params=True,
                opset_version=ONNX_OPT_SET,
                do_constant_folding=True,
                input_names = ['input_ids'],
                output_names = ['text_embeddings', 'pooler_output'],
                dynamic_axes={
                    'input_ids': {0: 'B'},
                    'text_embeddings': {0: 'B'}
                }
        )

    onnx_model = onnx.load(onnx_path)
    onnx_opt_model = optimize_clip(onnx_model)
    onnx.save(onnx_opt_model, onnx_opt_path)


def build_clip_engine():
    onnx_opt_path = os.path.join(ONNX_DIR, "clip.opt.onnx")
    engine_path = os.path.join(ENGINE_DIR, "clip.plan")
    build_engine(
        onnx_path=onnx_opt_path,
        engine_path=engine_path,
        input_profile={
            'input_ids': [(1, 77), (1, 77), (16, 77)]
        }
    )


##############
# unet
##############
def optimize_unet(onnx_graph, minimal_optimization=False):
    enable_optimization = not minimal_optimization

    # Decompose InstanceNormalization into primitive Ops
    bRemoveInstanceNorm = enable_optimization
    # Remove Cast Node to optimize Attention block
    bRemoveCastNode = enable_optimization
    # Remove parallel Swish ops
    bRemoveParallelSwish = enable_optimization
    # Adjust the bias to be the second input to the Add ops
    bAdjustAddNode = enable_optimization
    # Change Resize node to take size instead of scale
    bResizeFix = enable_optimization

    # Common override for disabling all plugins below
    bDisablePlugins = minimal_optimization
    # Use multi-head attention Plugin
    bMHAPlugin = True
    # Use multi-head cross attention Plugin
    bMHCAPlugin = True
    # Insert GroupNormalization Plugin
    bGroupNormPlugin = True
    # Insert LayerNormalization Plugin
    bLayerNormPlugin = True
    # Insert Split+GeLU Plugin
    bSplitGeLUPlugin = True
    # Replace BiasAdd+ResidualAdd+SeqLen2Spatial with plugin
    bSeqLen2SpatialPlugin = True

    opt = Optimizer(onnx_graph, verbose=False)
    opt.info('UNet: original')

    if bRemoveInstanceNorm:
        num_instancenorm_replaced = opt.decompose_instancenorms()
        opt.info('UNet: replaced '+str(num_instancenorm_replaced)+' InstanceNorms')

    if bRemoveCastNode:
        num_casts_removed = opt.remove_casts()
        opt.info('UNet: removed '+str(num_casts_removed)+' casts')

    if bRemoveParallelSwish:
        num_parallel_swish_removed = opt.remove_parallel_swish()
        opt.info('UNet: removed '+str(num_parallel_swish_removed)+' parallel swish ops')

    if bAdjustAddNode:
        num_adjust_add = opt.adjustAddNode()
        opt.info('UNet: adjusted '+str(num_adjust_add)+' adds')

    if bResizeFix:
        num_resize_fix = opt.resize_fix()
        opt.info('UNet: fixed '+str(num_resize_fix)+' resizes')

    opt.cleanup()
    opt.info('UNet: cleanup')
    opt.fold_constants()
    opt.info('UNet: fold constants')
    opt.infer_shapes()
    opt.info('UNet: shape inference')

    num_heads = 8
    if bMHAPlugin and not bDisablePlugins:
        num_fmha_inserted = opt.insert_fmha_plugin(num_heads)
        opt.info('UNet: inserted '+str(num_fmha_inserted)+' fMHA plugins')

    if bMHCAPlugin and not bDisablePlugins:
        props = cudart.cudaGetDeviceProperties(0)[1]
        sm = props.major * 10 + props.minor
        num_fmhca_inserted = opt.insert_fmhca_plugin(num_heads, sm)
        opt.info('UNet: inserted '+str(num_fmhca_inserted)+' fMHCA plugins')

    if bGroupNormPlugin and not bDisablePlugins:
        num_groupnorm_inserted = opt.insert_groupnorm_plugin()
        opt.info('UNet: inserted '+str(num_groupnorm_inserted)+' GroupNorm plugins')

    if bLayerNormPlugin and not bDisablePlugins:
        num_layernorm_inserted = opt.insert_layernorm_plugin()
        opt.info('UNet: inserted '+str(num_layernorm_inserted)+' LayerNorm plugins')

    if bSplitGeLUPlugin and not bDisablePlugins:
        num_splitgelu_inserted = opt.insert_splitgelu_plugin()
        opt.info('UNet: inserted '+str(num_splitgelu_inserted)+' SplitGeLU plugins')

    if bSeqLen2SpatialPlugin and not bDisablePlugins:
        num_seq2spatial_inserted = opt.insert_seq2spatial_plugin()
        opt.info('UNet: inserted '+str(num_seq2spatial_inserted)+' SeqLen2Spatial plugins')

    onnx_opt_graph = opt.cleanup(return_onnx=True)
    opt.info('UNet: final')
    return onnx_opt_graph


def export_unet_onnx():
    onnx_path = os.path.join(ONNX_DIR, "unet.onnx")
    onnx_opt_path = os.path.join(ONNX_DIR, "unet.opt.onnx")
    torch_model = UNet2DConditionModel.from_pretrained(
                _HF_UNET_MODEL_NAME,
                revision="fp16",
                torch_dtype=torch.float16,
        ).to(DEVICE)
    with torch.inference_mode(), torch.autocast(DEVICE):
        latent_height = IMAGE_HEIGHT // 8
        latent_width = IMAGE_WIDTH // 8
        inputs = (
            torch.randn(2*BATCH_SIZE, 4, latent_height, latent_width, dtype=torch.float32, device=DEVICE),
            torch.tensor([1.], dtype=torch.float32, device=DEVICE),
            torch.randn(2*BATCH_SIZE, TEXT_MAX_LENGHT, TEXT_EMBEDDING_SIZE, dtype=torch.float16, device=DEVICE),
        )
        torch.onnx.export(
                torch_model,
                inputs,
                onnx_path,
                export_params=True,
                opset_version=ONNX_OPT_SET,
                do_constant_folding=True,
                input_names = ['sample', 'timestep', 'encoder_hidden_states'],
                output_names = ['latent'],
                dynamic_axes={
                    'sample': {0: '2B', 2: 'H', 3: 'W'},
                    'encoder_hidden_states': {0: '2B'},
                    'latent': {0: '2B', 2: 'H', 3: 'W'}
                }
        )

    onnx_model = onnx.load(onnx_path)
    onnx_opt_model = optimize_unet(onnx_model)
    onnx.save(onnx_opt_model, onnx_opt_path)


def build_unet_engine():
    onnx_opt_path = os.path.join(ONNX_DIR, "unet.opt.onnx")
    engine_path = os.path.join(ENGINE_DIR, "unet_fp16.plan")
    build_engine(
        onnx_path=onnx_opt_path,
        engine_path=engine_path,
        input_profile={
            'sample': [(2, 4, 64, 64), (2, 4, 64, 64), (32, 4, 64, 64)],
            'encoder_hidden_states': [(2, 77, 768), (2, 77, 768), (32, 77, 768)]
        }
    )


##############
# vae decoder
##############
def optimize_vae(onnx_graph, minimal_optimization=False):
    enable_optimization = not minimal_optimization

    # Decompose InstanceNormalization into primitive Ops
    bRemoveInstanceNorm = enable_optimization
    # Remove Cast Node to optimize Attention block
    bRemoveCastNode = enable_optimization
    # Insert GroupNormalization Plugin
    bGroupNormPlugin = enable_optimization

    opt = Optimizer(onnx_graph, verbose=False)
    opt.info('VAE: original')

    if bRemoveInstanceNorm:
        num_instancenorm_replaced = opt.decompose_instancenorms()
        opt.info('VAE: replaced '+str(num_instancenorm_replaced)+' InstanceNorms')

    if bRemoveCastNode:
        num_casts_removed = opt.remove_casts()
        opt.info('VAE: removed '+str(num_casts_removed)+' casts')

    opt.cleanup()
    opt.info('VAE: cleanup')
    opt.fold_constants()
    opt.info('VAE: fold constants')
    opt.infer_shapes()
    opt.info('VAE: shape inference')

    if bGroupNormPlugin:
        num_groupnorm_inserted = opt.insert_groupnorm_plugin()
        opt.info('VAE: inserted '+str(num_groupnorm_inserted)+' GroupNorm plugins')

    onnx_opt_graph = opt.cleanup(return_onnx=True)
    opt.info('VAE: final')
    return onnx_opt_graph


def export_vae_onnx():
    onnx_path = os.path.join(ONNX_DIR, "vae.onnx")
    onnx_opt_path = os.path.join(ONNX_DIR, "vae.opt.onnx")
    torch_model = AutoencoderKL.from_pretrained(
            _HF_VAE_MODEL_NAME,
        ).to(DEVICE)
    torch_model.forward = torch_model.decode
    with torch.inference_mode(), torch.autocast(DEVICE):
        latent_height = IMAGE_HEIGHT // 8
        latent_width = IMAGE_WIDTH // 8
        inputs = torch.randn(BATCH_SIZE, 4, latent_height, latent_width, dtype=torch.float32, device=DEVICE)
        torch.onnx.export(
                torch_model,
                inputs,
                onnx_path,
                export_params=True,
                opset_version=ONNX_OPT_SET,
                do_constant_folding=True,
                input_names = ['latent'],
                output_names = ['images'],
                dynamic_axes={
                    'latent': {0: 'B', 2: 'H', 3: 'W'},
                    'images': {0: 'B', 2: '8H', 3: '8W'}
                }
        )

    onnx_model = onnx.load(onnx_path)
    onnx_opt_model = optimize_vae(onnx_model)
    onnx.save(onnx_opt_model, onnx_opt_path)


def build_vae_engine():
    onnx_opt_path = os.path.join(ONNX_DIR, "vae.opt.onnx")
    engine_path = os.path.join(ENGINE_DIR, "vae.plan")
    build_engine(
        onnx_path=onnx_opt_path,
        engine_path=engine_path,
        input_profile={
            'latent': [(1, 4, 64, 64), (1, 4, 64, 64), (16, 4, 64, 64)]
        }
    )


##############
# main
##############
if __name__ == "__main__":
    export_clip_onnx()
    export_unet_onnx()
    export_vae_onnx()

    build_clip_engine()
    build_unet_engine()
    build_vae_engine()
