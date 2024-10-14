

Codes for featurizing H&E Whole-Slide Images using Prov-GigaPath pre-trained model [1]


Step 1:

-Set up pyvips environment using environment_vips.yml:

  Hopefully a simple:
  
    conda env create -f environment_vips.yml
  
  should take care of the job.

Step 2:

-Tile the existing WSIs into .png patches to work with the simpler version of Prov-GigaPath (using OpenSlide would require more complex installation due to version issues with pixman)

  To run this, use the tiling.py script provided:

    python -u tiling.py --wsi_path "$wsi_path" --save_dir "$save_dir" --tile_size "$tile_size" --resize_factor "$resize_factor"

  Where the $wsi_path refers to a folder with all .tif WSIs, $save_dir refers to the folder where all the tiles will be saved, $tile_size is the tile size, and $resize_factor is the resize factor.


Step 3:

-Encode the tiled WSIs using Prov-GigaPath. This one will take about ~1 hour per slide using ~14GB of VRAM. 

  To run this step, first install the prov-gigapath repository and environment following their instructions (https://github.com/prov-gigapath/prov-gigapath/tree/main)

  Once this step is done, run the following:

    python -u encoding.py --tile_path "$tile_path" --save_dir "$encoding_save_dir"

  Where $tile_path = $save_dir from last step; and $encoding_save_dir is where the vectors will be saved


Once this script is done, send me the contents of $encoding_save_dir


  

