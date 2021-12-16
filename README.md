# frechet_distance
&nbsp;    &nbsp;    &nbsp;    &nbsp;    &nbsp;    看论文看到了这距离的作用，就试着学了一下，然后又恰巧发现这东西是个二维的dp 就用之前学的dp优化方法搞了一下。 
&nbsp;    &nbsp;    &nbsp;    &nbsp;    &nbsp;    这个距离的定义我就不讲了，数学描述烧脑子，有兴趣自己去看。直觉定义我放这儿。 

 > 一个人在遛狗，他们走在各自的道路上。他们可能有着不同的速度，但是都不能往回走。最终的目的，就是求满足要求的绳子的最小长度。

 # distance_lib.py
  
  &nbsp;    &nbsp;    &nbsp;    &nbsp;    &nbsp; 一个计算距离的库，目前其实就是把三种距离的计算打包在一个py文件里面，后续看你想用啥距离测度来描述两个点的距离（曼哈顿，欧式。切比雪夫、马氏、测地等等）可以直接添加到里面

# 大致内容
  &nbsp;    &nbsp;    &nbsp;    &nbsp;    &nbsp;    其实干了啥呢，原始的 `DiscreteFrechet` 是从后往前递归实现的，然后改成 dp 从前往后实现得到 `LinearDiscreteFrechet` ，然后对 dp 做了点优化减少了空间复杂度 得到`compressionLinearDiscreteFrechet`，后来又想着先把所有距离都算出来，利用查表法看能不能降低时间 `VectorizedDiscreteFrechet`，结果表现超级拉跨

 &nbsp;    &nbsp;    &nbsp;    &nbsp;    &nbsp;    这些优化其实都只是动态规划dp的知识，从递归改成dp体改了10倍左右速度，对dp进行优化只是缓解了空间复杂度，时间会多一点点，但可以接受


 # 结果


```python
Slow time and ferechet distance:
 0.02692690     83.23780104754721
Linear time and ferechet distance:
 0.00266960     83.23780104754721
VDF time and ferechet distance:
 0.21261920     83.23780104754721
compression_linear time and ferechet distance :
 0.00274860     83.23780104754721

time ratio(Based on linear):
Linear : compression_linear : :slow : VDA 
1.00 : 1.03 : 10.09 : 79.64 
```

