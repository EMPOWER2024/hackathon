import os
import argparse
import timm
from gigapath.pipeline import load_tile_slide_encoder
from gigapath.pipeline import run_inference_with_tile_encoder
from gigapath.pipeline import run_inference_with_slide_encoder


def main():
    parser = argparse.ArgumentParser(description="Preprocess parameters.")

    # Define arguments with their default values
    parser.add_argument("--tile_path", type=str, default='data/', help="Path to tile folder")
    parser.add_argument("--save_dir", type=str, default='proper_tiles/', help="Path to save WSI-level vectors")
    parser.add_argument("--hf_token", type=str, default='secret', help="Huggingface token")

    # Parse the arguments
    args = parser.parse_args()

    # Now you can use the arguments as variables in your code
    print(f"Tile Directory: {args.tile_path}")
    print(f"Save Directory: {args.save_dir}")


    TILE_PATH = args.tile_path
    SAVE_DIR = args.save_dir

    if not os.path.exists(TILE_PATH):
        print("Tile folder does not exist, script should be stopped now")
    else:
        if not os.path.exists(SAVE_DIR):
            print("Creating vector folder")
            os.makedirs(SAVE_DIR)


    # Please set your Hugging Face API token
    os.environ["HF_TOKEN"] = args.hf_token
    
    assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"


    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

    tile_encoder, slide_encoder_model = load_tile_slide_encoder(global_pool=True)

    for wsi in TILE_PATH:
        image_paths = [os.path.join(TILE_PATH+'/'+wsi, img) for img in os.listdir(wsi) if img.endswith('.png')]
        print(f"Found {len(image_paths)} image tiles")

        tile_encoder_outputs = run_inference_with_tile_encoder(image_paths, tile_encoder)

        for k in tile_encoder_outputs.keys():
            print(f"tile_encoder_outputs[{k}].shape: {tile_encoder_outputs[k].shape}")

        slide_embeds = run_inference_with_slide_encoder(slide_encoder_model=slide_encoder_model, **tile_encoder_outputs)
        print(slide_embeds.keys())

        for key in slide_embeds.keys():
            vector = np.asarray(slide_embeds[key])
            np.save(vector,SAVE_DIR+'/'+wsi+'_'+key+'.npy')
            

if __name__ == "__main__":
    main()
