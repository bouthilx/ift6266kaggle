=======================================================
transform : Extend dataset samples with transformations
=======================================================

Extends a dataset by adding random transformations

Usage
****
  python transform.py TRANSFORMATIONS DATASET NEW_DATASET 


Transformation file configuration example
****

::

        #name     settings
        translate ratio=5   mean=1    std=1   
        zoom      ratio=5   mean=1    std=0.1   
        rotate    ratio=5   mean=0    std=10    
        noise     ratio=0.5 sigma=1
        noise     ratio=0.5 sigma=2
        sharpen   ratio=0.5 sigma1=3  sigma2=1  alpha=30  
        denoise   ratio=0.5 sigma1=2  sigma2=3  alpha=0.4 
        clutter   ratio=5   mean=25   std=10.0  max_nb_per_image=2 
        flip      ratio=1

Examples
********

    python transformation.py transform.conf dataset.csv extended.csv
