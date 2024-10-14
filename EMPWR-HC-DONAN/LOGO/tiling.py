import pyvips as Vips
import shutil
import os
import argparse


def save_and_tile(image_to_segment, imagename, output_dir, tile_size=3072):
#     basename = os.path.basename(image_to_segment.filename)
#     base_dir_name = os.path.join(output_dir, basename.split('.svs')[0])
    base_dir_name = os.path.join(output_dir, imagename)
    print(base_dir_name)
    if not os.path.exists(base_dir_name):
        os.makedirs(base_dir_name)
    Vips.Image.dzsave(image_to_segment, base_dir_name,
                        layout='google',
                        suffix='.png',
                        tile_size=tile_size,
                        depth='one',
                        properties=True)
    return None


def main():
    parser = argparse.ArgumentParser(description="Preprocess parameters.")

    # Define arguments with their default values
    parser.add_argument("--wsi_path", type=str, default='data/', help="Path to WSI folder")
    parser.add_argument("--save_dir", type=str, default='proper_tiles/', help="Path to save  tiles")
    parser.add_argument("--tile_size", type=int, default=256, help="Tile sizes")
    parser.add_argument("--resize_factor", type=float, default=0.25, help="Resize factor for WSI")
    
    # Parse the arguments
    args = parser.parse_args()

    # Now you can use the arguments as variables in your code
    print(f"Tile Size: {args.tile_size}")
    print(f"WSI Directory: {args.wsi_path}")
    print(f"Save Directory: {args.save_dir}")
    print(f"Resize: {args.resize_factor}")


    WSI_PATH = args.wsi_path
    SAVE_DIR = 'temp_tiles/'
    store_path = args.save_dir
    TILE_SIZE = 256

    if not os.path.exists(WSI_PATH):
        print("WSI folder does not exist, script should be stopped now")
    else:
        if not os.path.exists(SAVE_DIR):
            print("Creating temporary tile folder")
            os.makedirs(SAVE_DIR)

        if not os.path.exists(store_path):
            print("Creating tile folder")
            os.makedirs(store_path)


    for wsi in sorted(os.listdir(WSI_PATH)):
        vips_img = Vips.Image.new_from_file(WSI_PATH+wsi, level=0)
        if args.resize_factor != 1:
            vips_img = vips_img.resize(args.resize_factor)
    
        save_and_tile(vips_img, wsi.split('.')[0], SAVE_DIR, tile_size = TILE_SIZE)
    
    
    for case_folder in sorted(os.listdir(SAVE_DIR)):
        NAID = case_folder
        print('Processing NAID: ', NAID)
        try:
            os.makedirs(store_path+NAID+'/images/')
        except:
            pass
        for tile_folder in sorted(os.listdir(SAVE_DIR+NAID+'/0/')):
            # folder_level == y axis distance determinant
            # file_level == x axis distance determinant
            y0 = int(tile_folder)*TILE_SIZE
            y1 = (int(tile_folder)+1)*TILE_SIZE
            for tile_file in sorted(os.listdir(SAVE_DIR+NAID+'/0/'+tile_folder+'/')):
                x0 = int(tile_file.split('.')[0])*TILE_SIZE
                x1 = (int(tile_file.split('.')[0])+1)*TILE_SIZE
                shutil.copy(SAVE_DIR+NAID+'/0/'+str(tile_folder)+'/'+tile_file, 
                          store_path+NAID+'/images/'+str(x0)+'x_'+str(y0)+'y.png')


    print("Deleting temporary folder")
    shutil.rmtree(SAVE_DIR)


if __name__ == "__main__":
    main()
