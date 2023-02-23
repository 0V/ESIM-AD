Event-based Camera Simulation using Monte Carlo Path Tracing with Adaptive Denoising
===

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />

Y. Tsuji, Y. Yatagawa, H. Kubo, S. Morishima, "Event-based Camera Simulation using Monte Carlo Path Tracing with Adaptive Denoising," 2023. [[Preprint]](https://drive.google.com/file/d/1JaQc_OJb3YTIAdd_N4Eaoq-EiHoDVwbL/view?usp=share_link)

## Install

#### Windows

```shell
# Install required modules
pipenv install --dev

# Enable Pipenv
pipenv shell
```

#### MacOS

```shell
# Install OpenEXR v2 (2.5.8) and IlmBase
brew install openexr@2
brew install ilmbase

# Path settings
export HOMEBREW_HOME=/opt/homebrew/Cellar
export CFLAGS="-std=c++11 -I$HOMEBREW_HOME/openexr@2/2.5.8/include/OpenEXR -I$HOMEBREW_HOME/ilmbase/2.5.8/include/OpenEXR"
export CPPFLAGS="-std=c++11 -I$HOMEBREW_HOME/openexr@2/2.5.8/include/OpenEXR -I$HOMEBREW_HOME/ilmbase/2.5.8/include/OpenEXR"
export LDFLAGS="-L$HOMEBREW_HOME/openexr@2/2.5.8/lib -L$HOMEBREW_HOME/ilmbase/2.5.8/lib"

# Install required modules
pipenv install --dev

# Enable Pipenv
pipenv shell
```

## Dataset

Download data from the following URL:  
<https://drive.google.com/file/d/1B8Juxd1llp__96dWlXDegZ-YuBRj27rc/view?usp=share_link>  
(Currently, only the "San Miguel" scene is shared, but we will share other data soon)

Then, unzip the archive and store in the `data` folder. If the data is `sanmiguel`, four subdirectories (i.e., `32spp`, `64spp`, `128spp`, and `4096spp`) will be stored in `data/sanmiguel`.

## Run

```shell
# To reproduce our results in the paper, run the following.
python main.py --data_root /path/to/data --spp 32 --ksize 13 --wlr simple
```

After running the above code, you can visually compare our method with other baselines with `analysis.ipynb` in the `notebooks` folder.

## Results

#### Living Room (32spp)

https://user-images.githubusercontent.com/1556493/220586376-3d1c804f-e168-4a9e-9e9c-140471a6b0d9.mp4

#### San Miguel (32spp)

https://user-images.githubusercontent.com/1556493/220586355-12b829dc-f8c9-4c94-9ab8-f3387594d31e.mp4

#### Two Boxes (32spp)

https://user-images.githubusercontent.com/1556493/220586327-585b55d2-6f51-46f3-93c7-4afa25e904a9.mp4

<br />

Note that the above videos are compressed to fit the 10MB file size limit of GitHub. You can find uncompressed ones in the following URL:  
<https://drive.google.com/file/d/1hPtCmBYmo7h-aczLSPVK8ZIK8_sUJ4nW/view?usp=share_link>

Also, you can find the quantitative comparisons in the supplementary document in the following URL:  
<https://drive.google.com/file/d/1gJsuApMHOO-PPWweZ849QaGdr0OQsjyl/view?usp=share_link>


## License

This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

(c) Yuta Tsuji, Tatsuya Yatagawa, Hiroyuki Kubo, and Shigeo Morishima.
