This repo contains examples that will be included in Legate's documentation (https://github.com/nv-legate)

Run `madelbrot.py` using different backends and see how easy it is to get performance improvements

To run using NumPy\
`python mandelbrot.py`

Use 8 OpenMP threads to compute the Mandelbrot set\
`LEGATE_TEST=1 legate --omps 1 --ompthreads 8 --regmem 10000 --sysmem 10000 ./mandelbrot.py`

Use 1 GPU to compute the Mandelbrot set\
`LEGATE_TEST=1 legate --gpus 1 --fbmem 10000 ./mandelbrot.py`

Use 4 CPUs threads to compute the Mandelbrot set.\
`LEGATE_TEST=1 legate --cpus 4 --sysmem 10000 ./mandelbrot.py`

(Link to FAQs on how to time, what these input arguments mean etc.)
