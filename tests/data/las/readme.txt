# Test LAS Data

I don't include a LAS file with this repo because they are often pretty large. Also because who knows what datum / format you want to use for your project, and it's far easier to use what you're planning to work with at test time than to assume things work for your datum (they are all over the place).

Anyway, take a rationally sized .las file that is representative of what you plan to work with, and copy it here. Name it `test_sample.las`. The .gitignore of this project will ignore it, but the test suite will be able to work with it, and will expect one here.

If you're seeing test suite failures due to a missing .las sample, this is what you need to do.

I have left some sample shp and shx and dbf files (metadata for these kinds of datasets) in this folder so the test suite can use those. They are from the NYC_2021 LiDAR dataset, which you can download from the NY GIS ftp site, if you want to just use what I was using when I made this program.