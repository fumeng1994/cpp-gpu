# cpp-gpu
C++ GPU Demo using Docker


| Docker                 |  | 
|:---------------------------------------------------------------------------|:-----------|
| docker login        |  should auto login | 
| docker build -t craftyapple/cpp-gpu:v1 . |  | 
| docker push craftyapple/cpp-gpu:v1          |  | 
| docker run --rm --gpus all craftyapple/cpp-gpu:v5 python3 -u handler.py 13 |  | 
