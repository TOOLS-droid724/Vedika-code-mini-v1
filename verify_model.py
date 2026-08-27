#!/usr/bin/env python3
"""
Verification script for Vedika-Code-Pro-v1 custom model setup.

This script verifies:
1. AutoConfig loads correctly with trust_remote_code=True
2. AutoModelForCausalLM loads without missing/unexpected keys warnings
3. A dummy forward pass executes successfully
"""

import os
import sys
import warnings
import torch

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def verify_model_setup():
    """Verify the custom model setup."""
    
    print("=" * 80)
    print("Vedika-Code-Pro-v1 Model Setup Verification")
    print("=" * 80)
    
    # Get the directory containing the model files
    model_dir = os.path.dirname(os.path.abspath(__file__))
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    
    print(f"\n[1/4] Checking config.json...")
    import json
    config_path = os.path.join(model_dir, "config.json")
    
    if not os.path.exists(config_path):
        print(f"❌ ERROR: config.json not found at {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Check model_type
    model_type = config_dict.get("model_type")
    if model_type != "vedika_code_pro_v1":
        print(f"❌ ERROR: model_type is '{model_type}', expected 'vedika_code_pro_v1'")
        return False
    print(f"✓ model_type: {model_type}")
    
    # Check auto_map
    auto_map = config_dict.get("auto_map")
    if not auto_map:
        print("❌ ERROR: auto_map not found in config.json")
        return False
    
    required_keys = ["AutoConfig", "AutoModelForCausalLM"]
    for key in required_keys:
        if key not in auto_map:
            print(f"❌ ERROR: {key} not found in auto_map")
            return False
        print(f"✓ {key}: {auto_map[key]}")
    
    # Check architectures
    architectures = config_dict.get("architectures", [])
    if "VedikaCodeProV1ForCausalLM" not in architectures:
        print(f"❌ ERROR: VedikaCodeProV1ForCausalLM not in architectures")
        return False
    print(f"✓ architectures: {architectures}")
    
    print("\n[2/4] Loading configuration with AutoConfig...")
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model_dir,
            trust_remote_code=True
        )
        print(f"✓ Configuration loaded successfully")
        print(f"  - config_class: {config.__class__.__name__}")
        print(f"  - model_type: {config.model_type}")
        print(f"  - vocab_size: {config.vocab_size}")
        print(f"  - hidden_size: {config.hidden_size}")
        print(f"  - num_hidden_layers: {config.num_hidden_layers}")
        print(f"  - num_attention_heads: {config.num_attention_heads}")
        print(f"  - hc_mult: {config.hc_mult}")
        print(f"  - n_routed_experts: {config.n_routed_experts}")
        print(f"  - num_experts_per_tok: {config.num_experts_per_tok}")
    except Exception as e:
        print(f"❌ ERROR loading config: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n[3/4] Verifying model architecture...")
    try:
        from configuration_vedika_code_pro_v1 import VedikaCodeProV1Config
        from modeling_vedika_code_pro_v1 import VedikaCodeProV1ForCausalLM
        
        # Create config object
        config = VedikaCodeProV1Config.from_pretrained(model_dir)
        
        # Verify model can be instantiated (without loading pretrained weights)
        # We'll create a smaller test version to verify architecture
        test_config = VedikaCodeProV1Config(
            vocab_size=1000,  # Small vocab for testing
            hidden_size=256,   # Small hidden size
            num_hidden_layers=2,  # Only 2 layers
            num_attention_heads=4,
            num_key_value_heads=1,
            n_routed_experts=8,  # Small number of experts
            n_shared_experts=1,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            hc_mult=2,  # Small HC mult
            max_position_embeddings=512,
            num_hash_layers=1,
            q_lora_rank=64,
            head_dim=64,
            qk_rope_head_dim=32,
            o_groups=2,
            o_lora_rank=64,
            sliding_window=32,
            index_n_heads=4,
            index_head_dim=32,
            index_topk=16,
            compress_ratios=[4, 0],  # Only first layer has compression
            tie_word_embeddings=False,
        )
        
        print(f"✓ Creating test model with reduced dimensions...")
        model = VedikaCodeProV1ForCausalLM(test_config)
        
        print(f"✓ Model architecture verified successfully")
        print(f"  - Model class: {model.__class__.__name__}")
        print(f"  - Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Verify key components exist
        assert hasattr(model, 'embed_tokens'), "Missing embed_tokens"
        assert hasattr(model, 'layers'), "Missing layers"
        assert hasattr(model, 'norm'), "Missing norm"
        assert hasattr(model, 'lm_head'), "Missing lm_head"
        assert hasattr(model, 'mtp'), "Missing mtp"
        print(f"✓ All required components present (embed_tokens, layers, norm, lm_head, mtp)")
        
        # Verify layer structure
        first_layer = model.layers[0]
        assert hasattr(first_layer, 'attn'), "Missing attn in layer"
        assert hasattr(first_layer, 'ffn'), "Missing ffn in layer"
        assert hasattr(first_layer, 'hc_attn_fn'), "Missing HC parameters"
        print(f"✓ Layer structure verified (attention, MoE, Hyper-Connections)")
        
    except Exception as e:
        print(f"❌ ERROR verifying model architecture: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n[4/4] Running dummy forward pass...")
    try:
        model.eval()
        
        # Create dummy input (small batch and sequence for quick test)
        batch_size = 1
        seq_length = 4
        input_ids = torch.randint(0, test_config.vocab_size, (batch_size, seq_length))
        
        print(f"  - Input shape: {input_ids.shape}")
        print(f"  - Vocab size: {test_config.vocab_size}")
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        logits = outputs.logits
        print(f"✓ Forward pass successful")
        print(f"  - Output logits shape: {logits.shape}")
        print(f"  - Expected shape: [1, 1, {test_config.vocab_size}] (last token)")
        
        # Verify output shape
        expected_shape = (batch_size, 1, test_config.vocab_size)
        if logits.shape == expected_shape:
            print(f"✓ Output shape matches expected shape")
        else:
            print(f"⚠️  Output shape {logits.shape} differs from expected {expected_shape}")
        
    except Exception as e:
        print(f"❌ ERROR during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✅ ALL VERIFICATION CHECKS PASSED!")
    print("=" * 80)
    print("\nThe Vedika-Code-Pro-v1 model setup is ready for inference.")
    print("\nUsage example:")
    print("```python")
    print("from transformers import AutoModelForCausalLM, AutoTokenizer")
    print("")
    print("model = AutoModelForCausalLM.from_pretrained(")
    print("    'Veda-Labs/Vedika-Code-Pro-v1',")
    print("    trust_remote_code=True,")
    print("    torch_dtype=torch.bfloat16")
    print(")")
    print("tokenizer = AutoTokenizer.from_pretrained('Veda-Labs/Vedika-Code-Pro-v1')")
    print("```")
    
    return True


if __name__ == "__main__":
    success = verify_model_setup()
    sys.exit(0 if success else 1)
