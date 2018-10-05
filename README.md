# Stochastic-flow

## やること
　[1]では、「ResNetの計算は常微分方程式の離散化」として、様々な考察を提案していた。  
　[2]では、さらに開き直り、常微分方程式の<img src="https://latex.codecogs.com/gif.latex?\frac{dx_t}{dt}=f(t,x_t)" />の右辺を直接近似すればいいという考えの元特殊なニューラルネットを構成していた。これでも精度は微減程度で済み、当然計算時間やメモリは大幅に圧縮されるのだという。  
　[3],[4]では、全結合層までの計算を輸送写像と考え、微分幾何や偏微分方程式論的なアプローチを展開している。  
　これらすべてで共通しているのは、「深層学習とは連続なflowの離散化である」という考え方だ。  
  今回行うのは、[1]でのSDEアプローチをさらに発展させるというもの。[1]ではStochastic DepthやShakeShake modelをSDEの簡易スキームによる近似と捉えていたわけだが、であればより良い離散近似を行えばより良いニューラルネットの構築ができることが期待される。  
  このレポジトリでは、[1]におけるStochastic Depthを簡易スキームだとした場合、この元となるSDEに対して様々な離散化を行い、それに対応するニューラルネットの性能を測る。
  


## モデル  
　用いるデータはCIFER10  
　
 * ResNet(通常実装)
 * Stochastic Depth(簡易スキーム)
 * Euler-丸山スキーム
 * ミルシュタインスキーム
 * あああああ
 
  この5種類を用いて数値実験を行う。
　下４つの手法に関しては、<img src="https://latex.codecogs.com/gif.latex?dX_t=p(t)f(X_t)dt+\sqrt{p(t)(1-p(t))}f(X_t)dB_t" />



# 参考文献
[1]



