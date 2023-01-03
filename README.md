# cog-karlo


[![Replicate](https://replicate.com/cjwbw/karlo/badge)](https://replicate.com/cjwbw/karlo) 


A Cog implementation for https://github.com/kakaobrain/karlo. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights 

Then, you can run predictions:

    cog predict -i prompt="a high-resolution photograph of a big red frog on a green leaf"
