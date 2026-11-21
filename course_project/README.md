sudo apt update && sudo apt install libopencv-dev

make cpu

./snake_app data/Lena.bmp --cpu

Optional snake tuning arguments:

```bash
./snake_app data/Lena.bmp --cpu \
  --alpha 2.0 --beta 1.5 --gamma 5.0 \
  --balloon 0.15 --window-radius 1 \
  --resample-interval 20 --resample-spacing 4.0 \
  --initial-spacing 10.0 --gaussian-sigma 1.0 \
  --iterations 500
```