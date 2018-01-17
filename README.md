# TorchKart

data
----
We prepare 10 races on Luigi Raceway for training at [data_link](work.hortune.tw:8000/60_data).
```
data/0_x.npy
data/0_y.npy
...
data/9_x.npy
data/9_y.npy
```
If you want to record your own data, run ```python3 record.py [id]``` will generate ```[id]_x.npy``` and ```[id]_y.npy```.

Training
----
```
python3 main.py --train
```
The program default load prepared 10 races ad training data, and save ```pre-train.pt``` for testing.

Testing
----
```
python3 main.py --test
```
The Program will load ```pre-train.pt``` and plays on Luigi Raceway.

## 環境架設
![](https://i.imgur.com/vp1qTXY.jpg)

在環境的架設上，我們會使用到mupen64plus，而網路上已經有人幫我們寫好他wrapper，然而這個wrapper有一些較嚴重的問題。

### python3 support
首先，此wrapper只支援python2而不支援python3，然而我們在training時會使用的pytorch卻只支援python3，因此我們trace了code並且做了一些相容性修正。

**修正一: HTTPServer**
在這份code中，他會架設一個http server用來當作command的傳輸，然而因為是python2所以是`from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer`，而在python3中`BaseHTTPServer`已經被替換成`http.server`。

**修正二: Encode 問題**
因為在python2的encode跟python3的encode剛好相反，因此在對socket傳輸時，必須加上`data.encode("utf8")`。

### 安裝環境
Ubuntu 16.04
python 3.5

### 安裝mupen64plus
`sudo apt install mupen64plus`
然後照著`gym_mupen64plus`安裝。 [link](https://github.com/hortune/gym_mupen64plus.git)
如果再安裝過程中有缺套件，請google一下，因為ubuntu的distribution上的套件最近更新頻繁，不太可能每個都列。
接著你需要重編你的cuda跟nvidia-driver，為了支援即時顯示。

```
# Download installers
mkdir ~/Downloads/nvidia
cd ~/Downloads/nvidia
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/384.59/NVIDIA-Linux-x86_64-384.59.run
sudo chmod +x NVIDIA-Linux-x86_64-384.59.run
sudo chmod +x cuda_8.0.61_375.26_linux-run
./cuda_8.0.61_375.26_linux-run -extract=~/Downloads/nvidia/
# Uninstall old stuff
sudo apt-get --purge remove nvidia-*
sudo nvidia-uninstall

# Reboot
sudo shutdown -r now
# In grub boot menu hit `e` and add `nouveau.modeset=0` to the end of the line beginning with `linux`
# F10 to boot
# CTRL+ALT+F1 and log in
sudo service lightdm stop
sudo ./NVIDIA-Linux-x86_64-384.59.run --no-opengl-files
sudo ./cuda-linux64-rel-8.0.61-21551265.run --no-opengl-libs
# Verify installation
nvidia-smi
cat /proc/driver/nvidia/version
```

重裝完後，你需要去下載vglrun，並且自己編譯並安裝。

**Warning**
如果你configure完vglrun，發現nvidia-smi permission不足，請手動修改顯卡所屬的user group。


## If You Hate The Setup
因為安裝真的很繁瑣，在這裡提供工作站一組帳密。
```
Username : ctf
Password : mlds
```

## After Installation If remote
接著，你必須在你的Windows電腦上安裝Vcxsrv或是XQuartz。 (如果是Vcxsrv 記得要把wg優化關掉)
如果沒有Windows電腦只能請你用204電腦implement。

連線時，必須要下 `ssh -XY username@work.hortune.tw`，沒下的話不會有X11 Forwarding，所以自己工作站的Forwarding也要打開。

然後應該就可以使用了。

## E.t.c
如果遇到任何困難，請email b04611015@csie.ntu.edu.tw，當初弄環境重灌了兩次電腦QQ，我不想被當QQ
