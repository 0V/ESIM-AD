Event-based Camera Simulation using Monte Carlo Path Tracing with Adaptive Denoising
===

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />

Y. Tsuji, Y. Yatagawa, H. Kubo, and S. Morishima, "Event-based Camera Simulation using Monte Carlo Path Tracing with Adaptive Denoising," In Proceedings of the IEEE International Conference on Image Processing, 2023. (to appear) [[Preprint]](https://drive.google.com/file/d/1JaQc_OJb3YTIAdd_N4Eaoq-EiHoDVwbL/view?usp=share_link)

## Install

Prepare virtualenv using [Poetry](https://python-poetry.org/).

```shell
# Install required modules
poetry install --no-dev

# Enable Poetry's virtualenv
poetry shell
```

## Dataset

- SanMiguel: <https://drive.google.com/file/d/1B8Juxd1llp__96dWlXDegZ-YuBRj27rc/view?usp=share_link>
- TwoBoxes: <https://drive.google.com/file/d/1XWpIgi9oR5SpYOAm_1CbnvixrfL42HdK/view?usp=drive_link>
- LivingRoom: https://drive.google.com/file/d/1prC6BAMmmqR0TJklhVbuiB3BnrlXJS4y/view?usp=drive_link

Then, unzip the archive and store in the `data` folder. If the data is `sanmiguel`, four subdirectories (i.e., `32spp`, `64spp`, `128spp`, and `4096spp`) will be stored in `data/sanmiguel`.

## Run

```shell
# To reproduce our results in the paper, run the following.
python main.py --data_root /path/to/data --spp 32 --ksize 13 --wlr simple --thres 1.00
```

The default parameters used to generate the following results are shown in `run.sh`. After running the above code, you can visually compare our method with other baselines with `analysis.ipynb` in the `notebooks` folder.

## Results

#### Living Room (32spp)

https://github.com/0V/ESIM-AD/assets/1556493/d684d4be-0970-4466-8531-1948db24a44d

#### San Miguel (32spp)

https://github.com/0V/ESIM-AD/assets/1556493/bfe0169c-1061-46ae-a1c3-2848ee410d3c

#### Two Boxes (32spp)

https://github.com/0V/ESIM-AD/assets/1556493/dd9594c9-46cf-4a75-975e-24a551c97a6d

<br />

Note that the above videos are compressed to fit the 10MB file size limit of GitHub. You can find uncompressed ones in the following URL:  
<https://drive.google.com/file/d/1hPtCmBYmo7h-aczLSPVK8ZIK8_sUJ4nW/view?usp=share_link>

Also, you can find the quantitative comparisons in the supplementary document in the following URL:  
<https://drive.google.com/file/d/1gJsuApMHOO-PPWweZ849QaGdr0OQsjyl/view?usp=share_link>


## License

This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

(c) Yuta Tsuji, Tatsuya Yatagawa, Hiroyuki Kubo, and Shigeo Morishima.
