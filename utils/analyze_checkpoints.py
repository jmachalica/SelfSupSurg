import torch
import pprint

def analyze_checkpoint(checkpoint_path, checkpoint_type="auto"):
    """
    Analyze a checkpoint and display its layer structure.
    
    Args:
        checkpoint_path: path to the checkpoint file
        checkpoint_type: "simclr", "imagenet", or "auto" (automatic detection)
    """
    print(f"\n🔍 Analyzing checkpoint: {checkpoint_path}")
    print("=" * 80)
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Automatically detect the type
        if checkpoint_type == "auto":
            if "classy_state_dict" in checkpoint:
                checkpoint_type = "simclr"
                print("🎯 Detected: VISSL/SimCLR checkpoint")
            elif isinstance(checkpoint, dict) and any("conv1" in k for k in checkpoint.keys()):
                checkpoint_type = "imagenet"
                print("🎯 Detected: ImageNet/torchvision checkpoint")
            else:
                checkpoint_type = "unknown"
                print("❓ Unknown checkpoint type")
        
        # Extract state_dict depending on checkpoint type
        if checkpoint_type == "simclr":
            state_dict = checkpoint.get("classy_state_dict", checkpoint)
        elif checkpoint_type == "imagenet":
            state_dict = checkpoint
        else:
            # Try common possible keys
            possible_keys = ["model", "state_dict", "classy_state_dict"]
            state_dict = None
            for key in possible_keys:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    print(f"📦 Using key: {key}")
                    break
            if state_dict is None:
                state_dict = checkpoint
        
        print(f"📊 Number of parameters: {len(state_dict)}\n")
        
        # Group layers by prefix
        layer_groups = {}
        for key in state_dict.keys():
            parts = key.split('.')
            if len(parts) > 1:
                prefix = parts[0]
                if prefix not in layer_groups:
                    layer_groups[prefix] = []
                layer_groups[prefix].append(key)
            else:
                layer_groups.setdefault("other", []).append(key)
        
        # Display grouped structure
        for group, layers in sorted(layer_groups.items()):
            print(f"📂 {group.upper()}:")
            for layer in sorted(layers):
                shape = list(state_dict[layer].shape) if hasattr(state_dict[layer], 'shape') else "N/A"
                print(f"   └── {layer}: {shape}")
            print()
        
        # Check for head layers
        head_layers = [k for k in state_dict.keys() if 'head' in k.lower()]
        if head_layers:
            print("🧠 HEAD LAYERS:")
            for layer in head_layers:
                shape = list(state_dict[layer].shape) if hasattr(state_dict[layer], 'shape') else "N/A"
                print(f"   └── {layer}: {shape}")
        else:
            print("❌ No HEAD layers found")
        
        # Check for trunk/backbone
        trunk_layers = [k for k in state_dict.keys() if any(x in k.lower() for x in ['trunk', 'backbone', 'feature_blocks'])]
        if trunk_layers:
            print("\n🌳 TRUNK/BACKBONE LAYERS (first 10):")
            for layer in sorted(trunk_layers)[:10]:
                shape = list(state_dict[layer].shape) if hasattr(state_dict[layer], 'shape') else "N/A"
                print(f"   └── {layer}: {shape}")
            if len(trunk_layers) > 10:
                print(f"   ... and {len(trunk_layers) - 10} more")
        
        return state_dict, checkpoint_type
        
    except Exception as e:
        print(f"❌ Error while loading: {e}")
        return None, None


def compare_checkpoints(checkpoint1_path, checkpoint2_path):
    """Compare two checkpoints."""
    print("\n🔄 CHECKPOINT COMPARISON")
    print("=" * 80)
    
    _, type1 = analyze_checkpoint(checkpoint1_path)
    _, type2 = analyze_checkpoint(checkpoint2_path)
    
    print(f"\n📋 SUMMARY:")
    print(f"Checkpoint 1 ({type1}): {checkpoint1_path}")
    print(f"Checkpoint 2 ({type2}): {checkpoint2_path}")


# Helper functions for specific cases
def analyze_simclr_checkpoint(checkpoint_path):
    """Dedicated analysis for SimCLR checkpoints."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("🔍 SIMCLR CHECKPOINT ANALYSIS")
    print("=" * 50)
    
    # Show available keys
    print("📦 Available keys in checkpoint:")
    for key in checkpoint.keys():
        print(f"   └── {key}")
    
    if "classy_state_dict" in checkpoint:
        state_dict = checkpoint["classy_state_dict"]
        print(f"\n🎯 Using 'classy_state_dict' ({len(state_dict)} parameters)")
        
        # Show projection heads (SimCLR)
        heads = [k for k in state_dict.keys() if k.startswith('heads')]
        print(f"\n🧠 PROJECTION HEADS ({len(heads)} layers):")
        for head in heads:
            shape = list(state_dict[head].shape)
            print(f"   └── {head}: {shape}")
    
    return checkpoint


def analyze_imagenet_checkpoint(checkpoint_path):
    """Dedicated analysis for ImageNet checkpoints."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("🔍 IMAGENET CHECKPOINT ANALYSIS")
    print("=" * 50)
    
    print(f"📊 Number of parameters: {len(checkpoint)}")
    
    # Key ResNet layers
    key_layers = ['conv1.weight', 'bn1.weight', 'layer1.0.conv1.weight', 'fc.weight', 'fc.bias']
    print(f"\n🔑 KEY LAYERS:")
    for layer in key_layers:
        if layer in checkpoint:
            shape = list(checkpoint[layer].shape)
            print(f"   ✅ {layer}: {shape}")
        else:
            print(f"   ❌ {layer}: MISSING")
    
    return checkpoint


# Example usage
if __name__ == "__main__":
    # simclr_path = "./checkpoints/defaults/resnet_50/simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1.torch"
    simclr_path = "/net/tscratch/people/plgjmachali/surgvu_results/pretraining/moco/job_1933181/model_final_checkpoint_phase1.torch"
    imagenet_path = "./checkpoints/defaults/resnet_50/resnet50-19c8e357.pth"

    print("🚀 FULL CHECKPOINT ANALYSIS")

    # 🔹 1️⃣ Analyze VISSL/SimCLR checkpoint
    analyze_checkpoint(simclr_path, "simclr")

    # 🔹 2️⃣ Analyze ImageNet checkpoint
    analyze_checkpoint(imagenet_path, "imagenet")

    # 🔹 3️⃣ Compare both
    compare_checkpoints(imagenet_path, simclr_path)

    # 🔹 4️⃣ Detailed SimCLR structure
    analyze_simclr_checkpoint(simclr_path)

    # 🔹 5️⃣ Detailed ImageNet structure
    analyze_imagenet_checkpoint(imagenet_path)