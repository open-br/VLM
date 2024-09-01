# super_ai

[![Apach2.0 License](http://img.shields.io/badge/license-Apache-blue.svg?style=flat)](LICENSE)

小型Vision and Langageモデルを利用した説明文生成アプリケーション。

以下のサイトで公開されているライブラリを使用しており、より使いやすい形で提供することを目的とする。
https://github.com/tosiyuki/LLaVA-JP

---

## Features

小型Vision and Langageモデルを利用した説明文生成アプリケーション

---

## Configuration
### Software
|  ライブラリ  |  バージョン  |
| ---- | ---- |
|  Python  |  3.11 |
|  CUDA  |  12.1.105 |
|  cudnn  |  9.1.0.70 |

### Hardware
|  ライブラリ  |  バージョン  |
| ---- | ---- |
|  OS  |  Ubuntu20.04  |
|  CPU   |  32GB  |
|  GPU   |  8GB [RTX2070SUPER]  |

---


## Setting
### Install
```
# 使用しているGPUに対応したnvidiaドライバをインストール
# 以下のサイトからダウンロード
# https://www.nvidia.co.jp/Download/index.aspx?lang=jp

# ライブラリのダウンロード
$ git clone https://github.com/open-br/VLM
$ cd VLM

# 環境構築
$ conda create -n llava_jp python=3.11
$ pip install accelerate==0.33.0
$ pip install transformers==4.44.2
$ pip install open-clip-torch==2.26.1
$ pip install einops==0.8.0

```
---
## Usage
```
# 説明文生成
Usage: python demo_llava.py [-g] [-p prompt] [-i image] [-t version]
-g: GPUフラグ
-p: 指示文
-i: 入力画像ファイル
-t: 指示文の前に入力される役割説明文のバージョン。v1, kara, testが存在する。詳細はllava/conversation.pyに記載。

$ python demo_llava.py -g -p この画像に写っている人の特徴を説明をしてください。 -i imgs/2men.jpg -t test
$ python demo_llava.py -g -p この画像には人が何人いますか？ -i imgs/2men.jpg
$ python demo_llava.py -g -p このロボットの色は？ -i imgs/body.jpg

```
---

## License
GitHub Changelog Generator is released under the [Apache License 2.0](https://opensource.org/licenses/Apache-2.0).

---

## Acknowledgements
https://github.com/tosiyuki/LLaVA-JP
---

## Feedback 
Any questions or suggestions?

You are welcome to discuss it on:

[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/dancing_nanachi)
---
