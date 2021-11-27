# 基于神经网络的歌声合成Demo 
- 详情请参考博客文章:https://errol.blog.csdn.net/article/details/121532637
- [歌声合成视频课程地址](https://edu.csdn.net/course/detail/31938)
1. 语言：日语
2. 数据集：[kiritan_singing database](https://zunko.jp/kiridev/login.php)
3. 输入musicxml file,输出音频。
4. [nnsvs](https://github.com/r9y9/nnsvs/)
5. https://github.com/r9y9/nnsvs/tree/master/egs/kiritan_singing.
6. 运行时间5分钟

## 提示
这个是demo版本，入门歌声合成学习。

## 下载 music xml 文件

```
$ git clone -q https://github.com/r9y9/kiritan_singing
```

## 安装要求

nnsvs 依赖 sinsy (C++ library) for the muxicxml to context feature conversion.

```
$ pip install -q -U numpy cython
$ rm -rf hts_engine_API sinsy pysinsy nnmnkwii nnsvs

# Binary dependencies
$ git clone -q https://github.com/r9y9/hts_engine_API
$ cd hts_engine_API/src && ./waf configure --prefix=/usr/ && sudo ./waf build > /dev/null 2>&1 && ./waf install
$ git clone -q https://github.com/r9y9/sinsy
$ cd sinsy/src/ && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/ .. && make -j > /dev/null 2>&1 && sudo make install
```
# Python dependencies
```
$ git clone -q https://github.com/r9y9/pysinsy
$ cd pysinsy && export SINSY_INSTALL_PREFIX=/usr/ && pip install -q .
$ git clone -q https://github.com/r9y9/nnmnkwii
$ cd nnmnkwii && pip install -q .
$ git clone -q https://github.com/r9y9/nnsvs
$ cd nnsvs && pip install -q .

```

## 下载预训练模型
```angular2
$ curl -q -LO https://www.dropbox.com/s/pctlausq00eecqp/20200502_kiritan_singing-00-svs-world.zip
$ unzip -qq -o 20200502_kiritan_singing-00-svs-world.zip
```
